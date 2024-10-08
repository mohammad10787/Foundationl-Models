#Use Multi-layer Perceptron for character based name generator
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# this is a test

words = open('data/names.txt', "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset
block_size = 3  #context length: how many characters do we take to predict the next one?

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


random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print(Xtr.shape, Ytr.shape) # dataset

emb_size = 10
num_layer1 = 250
dic_size = 27
batch_size = 32

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((dic_size, emb_size), generator=g)
W1 = torch.randn((emb_size * block_size, num_layer1), generator=g)
b1 = torch.randn(num_layer1, generator=g)
W2 = torch.randn((num_layer1, dic_size), generator=g)
b2 = torch.randn(dic_size, generator=g)
parameters = [C, W1, b1, W2, b2]

sum(p.nelement() for p in parameters) # number of parameters in total

for p in parameters:
  p.requires_grad = True

lri = []
lossi = []
stepi = []

print(C[Xtr].shape)

for i in range(200000):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))

  # forward pass
  emb = C[Xtr[ix]]
  h = torch.tanh(emb.view(-1, emb_size * block_size) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ytr[ix])

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  #lr = lrs[i]
  lr = 0.5 if i < 100000 else 0.05
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  lri.append(lr)
  stepi.append(i)
  lossi.append(loss.log10().item())

#print(loss.item())

emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, emb_size * block_size) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)
print(f"The training loss is: {loss}")

emb = C[Xdev]
h = torch.tanh(emb.view(-1, emb_size * block_size) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(f"The dev loss is: {loss}")

emb = C[Xte]
h = torch.tanh(emb.view(-1, emb_size * block_size) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yte)
print(f"The test loss is: {loss}")

plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

# training split, dev/validation split, test split
# 80%, 10%, 10%

context = [0] * block_size
C[torch.tensor([context])].shape

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

  out = []
  context = [0] * block_size  # initialize with all ...
  while True:
    emb = C[torch.tensor([context])]  # (1,block_size,d)
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break

  print(''.join(itos[i] for i in out))