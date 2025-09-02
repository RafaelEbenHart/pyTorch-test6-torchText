import requests
import torch
import string
import unicodedata
import torch.nn.functional as fn
from torch import nn
import time
import sys

device = "cpu" if torch.cuda.is_available else "cpu"

block_size = 64

book_dir = "data/song/songs.txt"

with open(book_dir,"r",encoding="utf-8") as f:
    text = f.read()
# print(text[:500])
chars = sorted(list(set(text)))
# print(chars) # mengurutkan character
allowed_letters = string.ascii_letters
vocabulary_size = len(chars)
# print(vocabulary_size)
# cek buku dan 500 elemen awal dari buku

def unicodeToAscii(s:str):
    return ''.join(
        c for c in unicodedata.normalize("NFD",s)
        if  unicodedata.category(c) != 'Mn'
        and c in allowed_letters
    )

# print(f"kata kapal_lawd to {unicodeToAscii('kapal_lawd')}")

# encode dan decode
string_to_int = {ch:i for i, ch in enumerate(chars) }
int_to_string = {i:ch for i , ch in enumerate(chars)}
encode = lambda s:[string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
# print(data)


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

class TinyCharModel(nn.Module):
    def __init__(self, vocab_size:int, embed_dim=128):
        super().__init__()
        self.embed= nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embed_dim)
        self.lstm= nn.LSTM(input_size=embed_dim,
                            hidden_size=256,
                            batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc= nn.Linear(in_features=256,
                                out_features=vocab_size)

    def forward(self,x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B*T,C)
        #     targets = targets.view(B*T)
        #     loss = fn.cross_entropy(logits, targets)
        return x

torch.manual_seed(42)
model = TinyCharModel(vocab_size=vocabulary_size)
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 500



for epoch in range(epochs):
    X,y = get_batch("train")
    logits = model(X)
    # print(logits.shape)
    B, T, C = logits.shape
    logits = logits.view(B*T,C)
    y = y.view(B * T)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        Xtest,yTest = get_batch("test")
        testLogits = model(Xtest)
        A, E, D = testLogits.shape
        testLogits = testLogits.view(A*E,D)
        yTest = yTest.view(A*E)
        testloss = loss_fn(testLogits,yTest)

    if epoch % 100 == 0 :
        print(f"Epoch: {epoch+100}/{epochs} | Train Loss: {loss.item():.5f} | Test Loss: {testloss.item():.5f}")

def generate(model,start=str,length = 250 ):
    model.eval()
    context = torch.tensor(encode(start), dtype=torch.long).unsqueeze(0)
    out = context.clone()

    for _ in range(length):
        logits= model(out[:, -block_size:])
        probs = fn.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs,num_samples=1)
        out = torch.cat([out,next_id], dim=1)

    return decode(out[0].tolist())

def text_show(text: str):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.15)
    print()
    return text
print("="*50)
test = "the same song"
generate_lyrics = generate(model=model,start=test)
text_show(generate_lyrics)

