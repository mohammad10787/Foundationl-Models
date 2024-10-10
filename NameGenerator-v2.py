#Use Multi-layer Perceptron for character based name generator
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import torch.nn as nn
from torch.optim import SGD

words = open('data/names.txt', "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3  # context length
emb_size = 10  # word embedding size
num_layer1 = 250  # num of nerons in layer 1
dic_size = 27  # dictionary size
batch_size = 32  # batch size

def build_dataset(words):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((dic_size, emb_size), generator=g)

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

model = nn.Sequential(
  nn.Linear(emb_size * block_size, num_layer1),
  nn.Tanh(),
  nn.Linear(num_layer1, dic_size)
)

loss = nn.CrossEntropyLoss()
opt = SGD(model.parameters(), lr = 0.01)

lossi = []
stepi = []

for i in range(200000):
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))
  emb = C[Xtr[ix]]
  emb = emb.view(-1, emb_size * block_size)
  opt.zero_grad()
  ypred = model(emb)
  loss_value = loss(ypred, Ytr[ix])
  loss_value.backward()
  opt.step()
  lossi.append(loss_value.item())
  stepi.append(i)

#print(lossi)
plt.plot(stepi, lossi)
plt.show()

emb = C[Xtr]
emb = emb.view(-1, emb_size * block_size)
logits = model(emb)
loss = F.cross_entropy(logits, Ytr)
print(f"The training loss is: {loss}")

emb = C[Xdev]
emb = emb.view(-1, emb_size * block_size)
logits = model(emb)
loss = F.cross_entropy(logits, Ydev)
print(f"The dev loss is: {loss}")

emb = C[Xte]
emb = emb.view(-1, emb_size * block_size)
logits = model(emb)
loss = F.cross_entropy(logits, Yte)
print(f"The test loss is: {loss}")


# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

  out = []
  context = [0] * block_size  # initialize with all ...
  while True:
    emb = C[torch.tensor([context])]  # (1,block_size,d)
    emb = emb.view(-1, emb_size * block_size)
    logits = model(emb)

    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break

  print(''.join(itos[i] for i in out))