import torch
import torch.nn as nn
from torch.nn import functional as F
import statistics
import re
import math
from collections import Counter
import random

# Hyperparameters
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

torch.manual_seed(1337)

# Load data
with open('C:/Users/M2-Winterfell/Downloads/ML/CSU44061-Machine-Learning/Final_Assignment_GPT/finalAssignment_musicDataset/MinjuanLuo/inputMelodiesAugmented_M.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def tokenize_melody(sequence):
    tokens = sequence.split(' ')
    return tokens

def detokenize_melody(tokens):
    return ' '.join(tokens)

tokens = tokenize_melody(text)
stoi = {ch: i for i, ch in enumerate(sorted(set(tokens)))}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

encode = lambda s: [stoi[token] for token in tokenize_melody(s)]
decode = lambda l: detokenize_melody([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def parse_token(token_str):
    token_str = token_str.strip()
    if token_str.startswith('[') and token_str.endswith(']'):
        content = token_str[1:-1]  # strip brackets
    else:
        return None, None
    content = re.sub(r'\s+', ' ', content).strip()
    parts = content.split(',')
    if len(parts) != 2:
        return None, None
    pitch = parts[0].strip().replace(' ', '')
    duration = parts[1].strip().replace(' ', '')
    return pitch, duration

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.rel_position = nn.Embedding(2 * block_size - 1, head_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, self.num_heads, T, self.head_size)
        k = self.key(x).view(B, self.num_heads, T, self.head_size)
        v = self.value(x).view(B, self.num_heads, T, self.head_size)
        rel_positions = torch.arange(T, dtype=torch.long, device=x.device)
        rel_positions = rel_positions[:, None] - rel_positions[None, :]
        rel_positions = rel_positions + block_size - 1
        rel_emb = self.rel_position(rel_positions)
        rel_emb = rel_emb.unsqueeze(0).unsqueeze(0).repeat(B, self.num_heads, 1, 1, 1)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        rel_scores = (q.unsqueeze(-2) * rel_emb).sum(-1)
        scores = scores + rel_scores
        scores = scores.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        context = scores @ v
        context = context.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)
        return self.proj(context)

class FeedForward(nn.Module):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = RelativeMultiHeadAttention(n_head, head_size)
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
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits_ = logits.view(-1, logits.size(-1))
            targets_ = targets.view(-1)
            loss = F.cross_entropy(logits_, targets_)
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens,
        avg_rest_duration=None, rest_duration_tolerance=1.5,
        avg_pitch_duration=None, pitch_duration_tolerance=1.5,
        repetition_penalty=1.1, duration_repetition_penalty=1.1,
        temperature=1.1, top_k=10
    ):
        def get_last_pitch_and_duration(idx_seq):
            if idx_seq.numel() == 0:
                return None, None
            last_token_str = itos[idx_seq[-1].item()]
            p, d = parse_token(last_token_str)
            return p, d

        def top_k_filter(logits, k):
            # keep only top-k tokens
            v, ix = torch.topk(logits, k)
            out = torch.full_like(logits, float('-inf'))
            out.scatter_(1, ix, v)
            return out

        last_pitch, last_duration = get_last_pitch_and_duration(idx[0])

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            # Apply repetition penalty for pitch
            if last_pitch is not None:
                for i_token in range(vocab_size):
                    token_str = itos[i_token]
                    p, d = parse_token(token_str)
                    if p == last_pitch and p is not None:
                        probs[0, i_token] /= repetition_penalty

            # Apply repetition penalty for duration
            if last_duration is not None:
                for i_token in range(vocab_size):
                    token_str = itos[i_token]
                    p, d = parse_token(token_str)
                    if d is not None and d == last_duration:
                        probs[0, i_token] /= duration_repetition_penalty

            # Apply rest duration penalty
            if avg_rest_duration is not None:
                for i_token in range(vocab_size):
                    token_str = itos[i_token]
                    p, d = parse_token(token_str)
                    if p == 'R' and d is not None:
                        try:
                            duration_val = float(d)
                            if duration_val > avg_rest_duration * rest_duration_tolerance:
                                probs[0, i_token] *= 0.1
                        except:
                            probs[0, i_token] *= 0.1

            # Apply pitch duration penalty (both lower and upper bounds)
            # Reintroduce upper bound for pitch durations as well
            if avg_pitch_duration is not None:
                for i_token in range(vocab_size):
                    token_str = itos[i_token]
                    p, d = parse_token(token_str)
                    if p is not None and p != 'R' and d is not None:
                        try:
                            duration_val = float(d)
                            lower_bound = avg_pitch_duration / pitch_duration_tolerance
                            upper_bound = avg_pitch_duration * pitch_duration_tolerance
                            # penalize durations outside this range
                            if duration_val < lower_bound or duration_val > upper_bound:
                                probs[0, i_token] *= 0.1
                        except:
                            probs[0, i_token] *= 0.1

            # Re-normalize after penalties
            probs = probs / probs.sum()

            # Apply top-k filtering
            logits = torch.log(probs)
            logits = top_k_filter(logits, top_k)
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            last_pitch, last_duration = get_last_pitch_and_duration(idx[0])

        return idx

# Compute average rest duration and pitch duration
rest_durations = []
pitch_durations = []
for token_id in train_data:
    token_str = itos[token_id.item()]
    p, d = parse_token(token_str)
    if p is not None and d is not None:
        try:
            val = float(d)
            if p == 'R':
                rest_durations.append(val)
            else:
                pitch_durations.append(val)
        except:
            pass

avg_rest_duration = statistics.mean(rest_durations) if rest_durations else 50.0
avg_pitch_duration = statistics.mean(pitch_durations) if pitch_durations else 50.0

model = GPTLanguageModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def perplexity(loss):
    return math.exp(loss)

def compute_unigram_distribution(data_tensor):
    counts = Counter(data_tensor.tolist())
    total = sum(counts.values())
    dist = {k: v/total for k,v in counts.items()}
    return dist

train_dist = compute_unigram_distribution(train_data)
val_dist = compute_unigram_distribution(val_data)

# Baseline: random model from train distribution
def sample_baseline_melody(length=500):
    items = list(train_dist.items())
    items.sort(key=lambda x: x[0])
    token_ids, probs_ = zip(*items)
    cumulative = []
    csum = 0.0
    for p in probs_:
        csum += p
        cumulative.append(csum)
    tokens_ = []
    for _ in range(length):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r < c:
                tokens_.append(token_ids[i])
                break
    return tokens_

def bigram_repetition_rate(tokens_list):
    if len(tokens_list) < 2:
        return 0.0
    repeat_count = sum(1 for i in range(1, len(tokens_list)) if tokens_list[i] == tokens_list[i-1])
    return repeat_count / (len(tokens_list) - 1)

def js_divergence(p, q):
    # Compute Jensen-Shannon divergence
    all_keys = set(p.keys()).union(set(q.keys()))
    p_vec = [p.get(k,0) for k in all_keys]
    q_vec = [q.get(k,0) for k in all_keys]
    p_vec = torch.tensor(p_vec, dtype=torch.float)
    q_vec = torch.tensor(q_vec, dtype=torch.float)
    m = 0.5*(p_vec+q_vec)
    kl_pm = F.kl_div(m.log(), p_vec, reduction='sum')
    kl_qm = F.kl_div(m.log(), q_vec, reduction='sum')
    js = 0.5*(kl_pm+kl_qm)
    return js.item()

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        train_ppl = perplexity(train_loss)
        val_ppl = perplexity(val_loss)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        print(f"         train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Final evaluation
final_losses = estimate_loss()
final_val_loss = final_losses['val']
final_val_ppl = perplexity(final_val_loss)
print(f"Final Model Val PPL: {final_val_ppl:.2f}")

# Generate model melody with improved diversity
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(
    context, max_new_tokens=500,
    avg_rest_duration=avg_rest_duration, rest_duration_tolerance=1.5,
    avg_pitch_duration=avg_pitch_duration, pitch_duration_tolerance=1.5,
    repetition_penalty=1.1, duration_repetition_penalty=1.1,
    temperature=1.1, top_k=10
)
model_melody_ids = generated[0].tolist()
model_melody_tokens = decode(model_melody_ids).split(' ')

# Save generated melody to file
lines = []
for i in range(0, len(model_melody_tokens), 10):
    line = ' '.join(model_melody_tokens[i:i+10])
    lines.append(line)
with open('generated_melody.txt', 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line + '\n')

print("Model generated melody (sample):", ' '.join(model_melody_tokens[:30]))

# Baseline melody
baseline_melody_ids = sample_baseline_melody(length=500)
baseline_melody_tokens = [itos[id_] for id_ in baseline_melody_ids]
print("Baseline random melody (sample):", ' '.join(baseline_melody_tokens[:30]))

baseline_ppl = float(vocab_size)  # Approx as uniform random

model_bigram_rep_rate = bigram_repetition_rate(model_melody_ids)
baseline_bigram_rep_rate = bigram_repetition_rate(baseline_melody_ids)

model_dist = Counter(model_melody_ids)
total_model = sum(model_dist.values())
model_dist = {k: v/total_model for k,v in model_dist.items()}

baseline_dist = Counter(baseline_melody_ids)
total_baseline = sum(baseline_dist.values())
baseline_dist = {k: v/total_baseline for k,v in baseline_dist.items()}

train_model_js = js_divergence(train_dist, model_dist)
train_baseline_js = js_divergence(train_dist, baseline_dist)

print("----- Final Evaluation Metrics -----")
print(f"Model Val PPL: {final_val_ppl:.2f}")
print(f"Baseline PPL (approx): {baseline_ppl:.2f}")
print(f"Model Bigram Rep Rate: {model_bigram_rep_rate:.2f}")
print(f"Baseline Bigram Rep Rate: {baseline_bigram_rep_rate:.2f}")
print(f"Train-Model JS Divergence: {train_model_js:.4f}")
print(f"Train-Baseline JS Divergence: {train_baseline_js:.4f}")
