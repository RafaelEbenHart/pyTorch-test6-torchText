import requests
import torch
import string
import unicodedata
import torch.nn.functional as fn
from torch import nn
import time
import sys

device = "cpu" if torch.cuda.is_available else "cpu"
book_dir = "data/song/songs.txt"
batch_size = 8
block_size = 64
max_iters = 1000
learning_rate = 1e-4
eval_iters = 100
n_embd = 128
n_head = 8
n_layer = 4
dropout = 0.2

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

class bigramLangaugeModel(nn.Module):
    def __init__(self, input:int, neuron:int, output:int):
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocabulary_size,n_embd)
        self.position_embeding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential()
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(in_features=n_embd,out_features=vocabulary_size)


        ## make a class block that can make layer as much as you want if neccesary
        # self.
        # self.