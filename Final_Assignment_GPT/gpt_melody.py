import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter
import math
import random

# Hyperparameters
batch_size = 64
block_size = 128
max_iters = 2000
eval_interval = 400
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 144
n_head = 4
n_layer = 4
dropout = 0.1

torch.manual_seed(1337)

# Load the melody dataset
with open('C:/Users/M2-Winterfell/Downloads/ML/CSU44061-Machine-Learning/Final_Assignment_GPT/finalAssignment_musicDataset/Giovanni/inputMelodiesAugmented.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Basic tokenization: treat each character as a token
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    source = train_data if split == 'train' else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    """Estimate train and val loss to measure perplexity."""
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
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_ppl = math.exp(losses['train'])
        val_ppl = math.exp(losses['val'])
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"         train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# After training, let's evaluate metrics:

# 1) Perplexity on validation
final_losses = estimate_loss()
model_val_ppl = math.exp(final_losses['val'])
print(f"Final Model Val PPL: {model_val_ppl:.2f}")

# 2) Generate a sample from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
model_generated_indices = model.generate(context, max_new_tokens=500)[0]
model_generated_melody = decode(model_generated_indices.tolist())

print("Model generated melody (sample):")
print(model_generated_melody)

# 3) Baseline: random generation according to training distribution
def get_training_distribution(data):
    text_str = decode(data.tolist())
    counts = Counter(text_str)
    total = sum(counts.values())
    dist = {k: v/total for k,v in counts.items()}
    return dist

train_dist = get_training_distribution(train_data)

def generate_random_baseline(dist, length=500):
    chars = list(dist.keys())
    weights = list(dist.values())
    return ''.join(random.choices(chars, weights=weights, k=length))

baseline_melody = generate_random_baseline(train_dist, length=500)
print("Baseline random melody (sample):")
print(baseline_melody)

# 4) Compute perplexity of baseline by scoring it with the model
#    We approximate by sliding over the baseline melody in blocks.
def score_sequence(seq):
    # Compute average loss over the sequence
    with torch.no_grad():
        enc = torch.tensor([stoi[c] for c in seq if c in stoi], dtype=torch.long, device=device)
        # Break into blocks
        total_loss = 0.0
        count = 0
        for i in range(0, len(enc)-block_size, block_size):
            x = enc[i:i+block_size].unsqueeze(0)
            y = enc[i+1:i+block_size+1].unsqueeze(0)
            _, l = model(x, y)
            total_loss += l.item()
            count += 1
        return math.exp(total_loss/count) if count > 0 else math.inf

baseline_ppl = score_sequence(baseline_melody)
print(f"Baseline Approx. PPL: {baseline_ppl:.2f}")

# 5) N-gram repetition (take bigrams for example)
def ngram_repetition_rate(melody, n=2):
    # Count how many times each n-gram appears and see the ratio of repeats
    grams = [melody[i:i+n] for i in range(len(melody)-n+1)]
    counts = Counter(grams)
    # Repetition rate: fraction of total n-grams that are in the top repeating sets.
    # Another approach: measure how many n-grams occur more than once.
    repeated = sum(1 for g,c in counts.items() if c > 1)
    return repeated / len(counts) if counts else 0

model_bigram_rep = ngram_repetition_rate(model_generated_melody, n=2)
baseline_bigram_rep = ngram_repetition_rate(baseline_melody, n=2)

print(f"Model bigram repetition rate: {model_bigram_rep:.2f}")
print(f"Baseline bigram repetition rate: {baseline_bigram_rep:.2f}")

# 6) Distribution comparison (Jensen-Shannon divergence)
def get_note_distribution_from_str(mel):
    c = Counter(mel)
    total = sum(c.values())
    return {k: v/total for k,v in c.items()}

def js_divergence(p, q):
    all_keys = set(p.keys()).union(q.keys())
    p_arr = [p.get(k,0) for k in all_keys]
    q_arr = [q.get(k,0) for k in all_keys]
    m = [(p_arr[i]+q_arr[i])/2 for i in range(len(all_keys))]

    def kl_divergence(a, b):
        return sum(a[i]*math.log(a[i]/b[i], 2) for i in range(len(a)) if a[i] > 0 and b[i] > 0)

    return 0.5*kl_divergence(p_arr,m) + 0.5*kl_divergence(q_arr,m)

model_dist = get_note_distribution_from_str(model_generated_melody)
baseline_dist = get_note_distribution_from_str(baseline_melody)

train_vs_model_jsd = js_divergence(train_dist, model_dist)
train_vs_baseline_jsd = js_divergence(train_dist, baseline_dist)

print(f"Train vs Model JS Divergence: {train_vs_model_jsd:.4f}")
print(f"Train vs Baseline JS Divergence: {train_vs_baseline_jsd:.4f}")

# Final outputs:
print("----- Final Evaluation Metrics -----")
print(f"Model Val PPL: {model_val_ppl:.2f}")
print(f"Baseline PPL (approx): {baseline_ppl:.2f}")
print(f"Model Bigram Rep Rate: {model_bigram_rep:.2f}")
print(f"Baseline Bigram Rep Rate: {baseline_bigram_rep:.2f}")
print(f"Train-Model JS Divergence: {train_vs_model_jsd:.4f}")
print(f"Train-Baseline JS Divergence: {train_vs_baseline_jsd:.4f}")
