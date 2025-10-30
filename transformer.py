import requests
import torch
import string
import unicodedata
import torch.nn.functional as f
from torch import nn
import time
import sys


device = "cuda" if torch.cuda.is_available() else "cpu"
book_dir = "data/another/chat_dataset_human_human.txt"
batch_size = 16  #  batch size
block_size = 128  #  context window
max_iters = 3000  #  iterasi training
learning_rate = 1e-4  # learning rate
test_iters = 100
n_embd = 256  #  embedding
n_head = 8
n_layer = 6  #  layer
dropout = 0.2

# baca dan parse menjadi pasangan (User1 -> User2)
with open(book_dir, "r", encoding="utf-8") as fh:
    raw = fh.read()

# blok dipisah oleh baris kosong; tiap blok mengandung baris User1 / User2
blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
pairs = []
for b in blocks:
    lines = [l.strip() for l in b.splitlines() if l.strip()]
    for i in range(len(lines)-1):
        if lines[i].startswith("User1:") and lines[i+1].startswith("User2:"):
            prompt = lines[i].split("User1:", 1)[1].strip()
            response = lines[i+1].split("User2:", 1)[1].strip()
            pairs.append((prompt, response))

# buat text training: setiap contoh = "User1: <prompt>\nUser2: <response>\n\n"
dataset_text = "\n\n".join([f"User1: {p}\nUser2: {r}" for p, r in pairs]) + "\n"

# build vocabulary dari dataset_text (char-level)
chars = sorted(list(set(dataset_text)))
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

vocabulary_size = len(chars)

# convert whole dataset to tensor for autoregressive training
data = torch.tensor(encode(dataset_text), dtype=torch.long)


# split train dan test data

n = int(0.8 * len (data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split:str, batch_size = 32):
    data_split = train_data if split == "train" else test_data
    ix = torch.randint(len(data_split) - block_size - 1, (batch_size,))
    X = torch.stack([data_split [i:i + block_size] for i in ix])
    y = torch.stack([data_split[i+1:i + block_size+1] for i in ix])
    return X,y
# train
X, y = get_batch("train")
Xtest, yTest = get_batch("test")
# print(X.shape, y.shape)
# print("Contoh:" , "".join([int_to_string[i.item()] for i in X[0]]))
# print("Contoh:" , "".join([int_to_string[i.item()] for i in y[0]]))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class Head(nn.Module):
    """head dari multihead attention"""
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)   # (B, T, T)
        mask = self.tril[:T, :T]                               # (T, T)
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = f.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v                                          # (B, T, head_size)
        return out

class MultiheadAttention(nn.Module):
    """ multi head attention"""
    def __init__(self, n_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ layer pada llm"""
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

class GPTArch(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embeding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embeding_table(index)                     # (B, T, C)
        pos_idx = torch.arange(T, device=device)
        pos_emb = self.position_embeding_table(pos_idx).unsqueeze(0)   # (1, T, C)
        x = tok_emb + pos_emb                                          # (B, T, C)
        x = self.blocks(x)                                             # (B, T, C)
        x = self.ln_f(x)                                               # (B, T, C)
        logits = self.lm_head(x)                                       # (B, T, vocab)

        if targets is None:
            return logits, None

        logits_flat = logits.view(B * T, -1)
        targets_flat = targets.view(B * T)
        loss = f.cross_entropy(logits_flat, targets_flat)
        return logits, loss

# contoh training loop
# model = GPTArch(vocabulary_size).to(device)
# model.apply(model._init_weights)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
# for it in range(max_iters):
#     model.train()
#     X, y = get_batch("train", batch_size)
#     X = X.to(device); y = y.to(device)
#
#     logits, loss = model(X, y)   # forward mengembalikan loss jika targets diberikan
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if it % test_iters == 0:
#         model.eval()
#         with torch.inference_mode():
#             Xv, yv = get_batch("test", batch_size)
#             Xv = Xv.to(device); yv = yv.to(device)
#             _, test_loss = model(Xv, yv)
#         print(f"iter {it}/{max_iters} train_loss={loss.item():.4f} test_loss={test_loss.item():.4f}")
#
# # letakkan setelah training loop selesai
# checkpoint = {
#     'model_class': 'GPTArch',
#     'model_args': {
#         'vocab_size': vocabulary_size,
#         'n_embd': n_embd,
#         'n_head': n_head,
#         'n_layer': n_layer,
#         'block_size': block_size,
#         'dropout': dropout,
#     },
#     'model_state_dict': model.state_dict(),
#     'string_to_int': string_to_int,
#     'int_to_string': int_to_string,
# }
# torch.save(checkpoint, 'model_checkpoint.pth')
# print("Checkpoint saved -> model_checkpoint.pth")

if __name__ == "__main__":
    # buat model dan training hanya ketika file ini dijalankan langsung
    model = GPTArch(vocabulary_size).to(device)
    model.apply(model._init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(max_iters):
        model.train()
        X, y = get_batch("train", batch_size)
        X = X.to(device); y = y.to(device)

        logits, loss = model(X, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % test_iters == 0:
            model.eval()
            with torch.inference_mode():
                Xv, yv = get_batch("test", batch_size)
                Xv = Xv.to(device); yv = yv.to(device)
                _, test_loss = model(Xv, yv)
            print(f"iter {it}/{max_iters} train_loss={loss.item():.4f} test_loss={test_loss.item():.4f}")

    # save checkpoint and optional full model
    checkpoint = {
        'model_class': 'GPTArch',
        'model_args': {
            'vocab_size': vocabulary_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'dropout': dropout,
        },
        'model_state_dict': model.state_dict(),
        'string_to_int': string_to_int,
        'int_to_string': int_to_string,
    }
    torch.save(checkpoint, 'model_checkpoint.pth')
    # torch.save(model, 'model_full.pth')  # opsional: agar inference tidak perlu class
    print("Checkpoint saved -> model_checkpoint.pth")







