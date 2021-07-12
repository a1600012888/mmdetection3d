import torch

x = torch.tensor([[0, 1], [2, 0]])
y = !(x > 0)
z = torch.tensor([[2, 2], [2, 2]])
z = z * y
print(x)
print(y)
print(z)