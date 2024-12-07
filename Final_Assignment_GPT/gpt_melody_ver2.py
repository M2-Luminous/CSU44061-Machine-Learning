import torch
import torch.nn as nn
from torch.nn import functional as F
import statistics
import re

# Hyperparameters
batch_size = 64
block_size = 128
max_iters = 1000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 144
n_head = 4
n_layer = 4
dropout = 0.2

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
        out[split] = losses.mean()
    model.train()
    return out

# Parsing function to handle extra spaces
def parse_token(token_str):
    # Expected format: "[Pitch,Duration]"
    token_str = token_str.strip()
    if token_str.startswith('[') and token_str.endswith(']'):
        content = token_str[1:-1]  # strip brackets
    else:
        return None, None

    # Normalize spaces
    content = re.sub(r'\s+', ' ', content).strip()
    parts = content.split(',')
    if len(parts) != 2:
        return None, None

    pitch = parts[0].strip()
    duration = parts[1].strip()

    # Remove internal spaces
    pitch = pitch.replace(' ', '')
    duration = duration.replace(' ', '')

    return pitch, duration

class RelativeMultiHeadAttention(nn.Module):
    """Self-attention with relative position embeddings."""
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
        rel_emb = self.rel_position(rel_positions)  # (T, T, head_size)

        rel_emb = rel_emb.unsqueeze(0).unsqueeze(0)  
        rel_emb = rel_emb.repeat(B, self.num_heads, 1, 1, 1)

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
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, 
        avg_rest_duration=None, rest_duration_tolerance=1.5,
        avg_pitch_duration=None, pitch_duration_tolerance=1.5,
        repetition_penalty=1.1, duration_repetition_penalty=1.05
    ):
        def get_last_pitch_and_duration(idx_seq):
            if idx_seq.numel() == 0:
                return None, None
            last_token_str = itos[idx_seq[-1].item()]
            p, d = parse_token(last_token_str)
            return p, d

        last_pitch, last_duration = get_last_pitch_and_duration(idx[0])

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
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
                        # Penalize repeating the exact same duration token
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

            # Apply pitch duration penalty
            if avg_pitch_duration is not None:
                for i_token in range(vocab_size):
                    token_str = itos[i_token]
                    p, d = parse_token(token_str)
                    if p is not None and p != 'R' and d is not None:
                        try:
                            duration_val = float(d)
                            lower_bound = avg_pitch_duration / pitch_duration_tolerance
                            upper_bound = avg_pitch_duration * pitch_duration_tolerance
                            if duration_val < lower_bound or duration_val > upper_bound:
                                probs[0, i_token] *= 0.1
                        except:
                            probs[0, i_token] *= 0.1

            # Re-normalize probabilities
            probs = probs / probs.sum()

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            last_pitch, last_duration = get_last_pitch_and_duration(idx[0])

        return idx

# Compute average rest duration and pitch duration from training data
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

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(
    context, max_new_tokens=500,
    avg_rest_duration=avg_rest_duration, rest_duration_tolerance=1.5,
    avg_pitch_duration=avg_pitch_duration, pitch_duration_tolerance=1.5,
    repetition_penalty=1.1, duration_repetition_penalty=1.05
)
generated_tokens = decode(generated[0].tolist())

# Format and save the generated melody to a txt file
# Each line contains 10 tokens
tokens_list = generated_tokens.split(' ')
lines = []
for i in range(0, len(tokens_list), 10):
    line = ' '.join(tokens_list[i:i+10])
    lines.append(line)

with open('generated_melody.txt', 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line + '\n')

# Print the final generated melody
print(detokenize_melody(tokens_list))
