import math
import time
import datetime
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
batch_size = 32
context_size = 2048
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
embedding_size = 128
epochs = 100
learning_rate = 1e-3
eval_intervals = 10
num_heads = 8

# Load the data
with open('./data/test_data.txt', 'r') as f:
    text = f.read()
vocab = sorted(list(set(text))) + ['<PAD>', '<SOS>', '<EOS>', '<UNK>']  # Assuming the tokens are character level, we can also use word level tokenization using libraries
vocab_size = len(vocab)  # This is vocab of the model

# Create a mapping
str_to_int = {ch: i for i, ch in enumerate(vocab)}
int_to_str = {i: ch for i, ch in enumerate(vocab)}
encode = lambda x: [str_to_int[c] for c in x]  # Encoder to convert tokens from string to integer values
decode = lambda x: ''.join([int_to_str[c] for c in x])  # Decoder to convert tokens from integer to string values

# Split the data into train and validation
data = torch.tensor(encode(text), dtype=torch.long, device=device)
train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)


# Get random batches for training or testing
def get_batches(split):
    df = train_data if split == 'train' else val_data
    idx = torch.randint(len(df) - context_size - 1, (batch_size,))  # Get random indices from data for training
    x = torch.stack([df[i:i+context_size] for i in idx]).long()
    y = torch.stack([df[i+1:i+context_size+1] for i in idx]).long()
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batches(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


class RoPEAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(embedding_size, embedding_size, bias=False)
        self.query = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value = nn.Linear(embedding_size, embedding_size, bias=False)
        self.multihead = nn.MultiheadAttention(embedding_size, num_heads, dropout=0.1, batch_first=True, device=device)
        self.R = self.get_rotary_matrix(context_size, embedding_size)

    @staticmethod
    def get_rotary_matrix(context, embed):
        R = torch.zeros((context, embed, embed), requires_grad=False, device=device)
        for pos in range(context):
            for i in range(embed // 2):
                theta = pos * math.pow(10000, -2 * (i - 1) / embed)
                R[pos, 2 * i, 2 * i] = np.cos(theta)
                R[pos, 2 * i, 2 * i + 1] = - np.sin(theta)
                R[pos, 2 * i + 1, 2 * i] = np.sin(theta)
                R[pos, 2 * i + 1, 2 * i + 1] = np.cos(theta)

        return R

    def forward(self, x, return_attention_weights=False):
        B, M, D = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        q_out = (torch.bmm(Q.transpose(0,1), self.R[:M, :, :])).transpose(0,1)
        k_out = (torch.bmm(K.transpose(0, 1), self.R[:M, :, :])).transpose(0, 1)
        v_out = (torch.bmm(V.transpose(0, 1), self.R[:M, :, :])).transpose(0, 1)

        activations, attention_weights = self.multihead(
            q_out, k_out, v_out,
            attn_mask=nn.Transformer.generate_square_subsequent_mask(M).to(device),
            is_causal=False
        )

        if return_attention_weights:
            return activations, attention_weights

        return activations
        

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        ff_rms = torch.linalg.norm(x, dim=(1, 2)) * x[0].numel() ** -0.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)

        return self.scale[:x.shape[1], :].unsqueeze(0) * raw


class Llama_LLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(len(vocab), embedding_size)
        self.rms = RMSNorm((context_size, embedding_size))
        self.rope_attention = RoPEAttention()
        self.linear = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
        )
        self.out = nn.Linear(embedding_size, len(vocab))

    def forward(self, idx, targets=None):
        loss = None
        x = self.token_embedding(idx)
        x = self.rms(x)
        x = x + self.rope_attention(x)
        x = self.rms(x)
        x = x + self.linear(x)
        x = self.out(x)
        logits = x #F.softmax(x, dim=-1)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, len(vocab)), targets.view(-1))
            return logits, loss

        else:
            return logits

    def generate(self, idx, max_token=2048):
        for _ in range(max_token):
            logits = self(idx[:, -context_size:])
            # Only getting the last prediction from context. We can also sample from the distribution
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx


losses = []
start_time = time.time()
model = Llama_LLM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = None #torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    optimizer.zero_grad()
    x, y = get_batches('train')
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

    if scheduler:
        scheduler.step()

    if epoch % eval_intervals == 0:
        batch_time = time.time() - start_time
        loss = estimate_loss()
        losses += [loss]
        print(f'Epoch: {epoch}, Train Loss: {loss["train"]:.4f}, Val Loss: {loss["val"]:.4f}, Time: {batch_time:.2f}' +
             f" | ETA in seconds {batch_time * (epochs - epoch)/eval_intervals :.3f}")
    
        if scheduler:
            print("lr: ", scheduler.get_lr())

print("validation loss: ", losses[-1]['val'])
res = pd.DataFrame.from_records(losses)
res.train = res.train.apply(lambda x: x.detach().numpy())
res.val = res.val.apply(lambda x: x.detach().numpy())

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(res.index, res.train, 'b', label='Training loss')
plt.plot(res.index, res.val, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Saving the figure
output_dir = './results'  # Replace with your desired folder path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, f'train_val_loss{str(datetime.datetime.now().strftime("%m_%d_%Y, %H_%M_%S"))}.png'))
plt.close()  # Close the figure to free memory

# Generate a sample output
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_token=100)[0].tolist()))

