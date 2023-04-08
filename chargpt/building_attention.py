# testing code here
import torch
from torch.nn import Linear
from torch.nn import functional as F

B, T, C = 4, 8, 2

x = torch.randn(B, T, C)
print(x.shape)
# print(x)

# goal is to average over previous time steps
# output should be B, N, C where N is number of time steps - 1  ie T-1

# simple version - then vectorise
xav = torch.zeros(B, T, C)
for batch in range(B):
    for time in range(T):
        xav[batch, time] = torch.mean(x[batch, :time + 1], dim=0)
print("Basic version")
print(xav[0])

# vectorized (mat mult)
sum_mat = torch.tril(torch.ones(T, T))
sum_mat = sum_mat / torch.sum(sum_mat, dim=1, keepdim=True)
res = sum_mat @ x  # (T, T) @ (B, T, C) -> (B, T, C)
print("Matrix mult version")
print(res[0])

# self attention precursor version
tril = torch.tril(torch.ones(T, T))
# these are initial weights (or could be random)
weights = torch.zeros(T, T)
# this clamps the future to be inaccessible to the past
weights = torch.masked_fill(weights, tril == 0, float("-inf"))
# dim=-1 makes it row-wise
weights = F.softmax(weights, dim=-1)
res = weights @ x
print("Self Attention precursor")
print(res[0])

# basic self attention version
# the idea is to have Queries and Keys and the agreement between them will become the weights.
tril = torch.tril(torch.ones(T, T))

head_size = 8
query = Linear(C, head_size)
key = Linear(C, head_size)
# we add a value here - can be thought of as public information to transmit
# see fast Transformer paper - seems unnecessary (like GRU vs LSTM maybe)
value = Linear(C, head_size)

q = query(x)  # (B, T, C) -> (B, T, head_size)
k = key(x)  # (B, T, C) -> (B, T, head_size)
v = value(x)  # (B, T, C) -> (B, T, head_size)

weights = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
weights = weights * head_size ** -0.5
weights = torch.masked_fill(weights, tril == 0, float("-inf"))
weights = F.softmax(weights, dim=-1)

res = weights @ v
print(f"Weights\n {weights}")
print("Self Attention")
print(res[0])
