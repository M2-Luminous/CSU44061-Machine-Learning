import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from collections import Counter
import random
import numpy as np
import scipy.stats

# hyperparameters
batch_size = 64
block_size = 128
max_iters = 2000
eval_interval = 400
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 144
n_head = 4
n_layer = 4
dropout = 0.1

# Additional hyperparameters for time-aware and segmented attention
time_decay_scale = 50.0  # the scale factor for time decay in attention
window_size = 128        # window size for segmented attention

torch.manual_seed(1337)

# Read and parse the durations file
with open('C:/Users/M2-Winterfell/Downloads/ML/CSU44061-Machine-Learning/Final_Assignment_GPT/finalAssignment_musicDataset/MinjuanLuo/inputMelodiesAugmented_D.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

# Parse into integers assuming format: [0] [73] [73] ...
tokens_str = text.replace('[','').replace(']','').split()
tokens = [int(x) for x in tokens_str]

unique_tokens = sorted(list(set(tokens)))
vocab_size = len(unique_tokens)

stoi = {tok:i for i,tok in enumerate(unique_tokens)}
itos = {i:tok for tok,i in stoi.items()}

encode = lambda s: [stoi[x] for x in s]
decode = lambda l: [itos[i] for i in l]

data = torch.tensor(encode(tokens), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    """ One head of self-attention with time-aware and windowed mechanism. """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # Compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,T)

        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Time-aware attention: decay weights based on temporal distance
        # dist[i,j] = |i-j|
        dist = torch.arange(T, device=x.device).unsqueeze(0) - torch.arange(T, device=x.device).unsqueeze(1)
        dist = dist.abs().unsqueeze(0) # (1,T,T)
        
        # Apply exponential decay: closer tokens have less penalty
        time_decay = torch.exp(-dist / time_decay_scale)
        wei = wei + torch.log(time_decay + 1e-9)  # Adding in log-space for stability

        # Segmented/windowed attention: mask out tokens outside window_size
        # For token at position i, only attend to [i-window_size, i]
        # dist > window_size means too far
        wei = wei.masked_fill(dist > window_size, float('-inf'))

        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)

        v = self.value(x) # (B,T,hs)
        out = wei @ v      # (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple feedforward layer """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (FFN) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def compute_ppl(loss):
    return math.exp(loss)

# Build train distribution (index-based)
train_counts = Counter(train_data.tolist())
train_total = sum(train_counts.values())
train_dist = {k: v/train_total for k,v in train_counts.items()}

def random_baseline_ppl():
    val_tokens = val_data.tolist()
    nll = 0.0
    for t in val_tokens:
        p = train_dist.get(t, 1e-9)
        nll += -math.log(p+1e-9)
    return math.exp(nll / len(val_tokens))

def bigram_repetition_rate(seq):
    bigrams = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
    bigram_count = Counter(bigrams)
    repeated = sum(1 for b,cnt in bigram_count.items() if cnt > 1)
    return repeated / len(bigrams) if len(bigrams) > 0 else 0.0

def js_divergence(p, q):
    P = np.array(p, dtype=np.float64)
    Q = np.array(q, dtype=np.float64)
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5*(P+Q)
    return 0.5*(scipy.stats.entropy(P,M) + scipy.stats.entropy(Q,M))

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        train_ppl = compute_ppl(train_loss)
        val_ppl = compute_ppl(val_loss)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        print(f"         train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final evaluation
losses = estimate_loss()
val_ppl = compute_ppl(losses['val'])

# Generate a model sample
context = torch.zeros((1,1), dtype=torch.long, device=device)
model_out = model.generate(context, max_new_tokens=10)
model_seq = decode(model_out[0].tolist())

# Baseline metrics
baseline_ppl = random_baseline_ppl()

model_bigram_rep = bigram_repetition_rate(model_out[0].tolist())
baseline_sample_indices = np.random.choice(range(vocab_size), size=10, p=np.array([train_dist[i] for i in range(vocab_size)]))
baseline_bigram_rep = bigram_repetition_rate(baseline_sample_indices.tolist())
baseline_sample = decode(baseline_sample_indices.tolist())

with torch.no_grad():
    model_long = model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=10000)[0].tolist()

model_counts = Counter(model_long)
model_dist = np.array([model_counts[i]/len(model_long) if i in model_counts else 0 for i in range(vocab_size)])
train_dist_array = np.array([train_dist[i] for i in range(vocab_size)])
model_js = js_divergence(train_dist_array, model_dist)

baseline_js = js_divergence(train_dist_array, train_dist_array)

print(f"Final Model Val PPL: {val_ppl:.2f}")
print("Model generated (sample):")
print(model_seq)
print("Baseline random (sample):")
print(baseline_sample)

print(f"Baseline Approx. PPL: {baseline_ppl:.2f}")
print(f"Model bigram repetition rate: {model_bigram_rep:.2f}")
print(f"Baseline bigram repetition rate: {baseline_bigram_rep:.2f}")
print(f"Train vs Model JS Divergence: {model_js:.4f}")
print(f"Train vs Baseline JS Divergence: {baseline_js:.4f}")

print("----- Final Evaluation Metrics -----")
print(f"Model Val PPL: {val_ppl:.2f}")
print(f"Baseline PPL (approx): {baseline_ppl:.2f}")
print(f"Model Bigram Rep Rate: {model_bigram_rep:.2f}")
print(f"Baseline Bigram Rep Rate: {baseline_bigram_rep:.2f}")
print(f"Train-Model JS Divergence: {model_js:.4f}")
print(f"Train-Baseline JS Divergence: {baseline_js:.4f}")
