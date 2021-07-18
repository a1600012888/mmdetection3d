import torch

query = torch.randn(300, 1, 256)
value = torch.randn(184950, 1, 256)
key = query

attn = torch.nn.MultiheadAttention(256, 8)

out = attn(query=query, key=key, value=value)
print(out[0].size())