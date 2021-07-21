import torch

query = torch.randn(300, 1, 256)
value = key = query
query_pos = torch.randn(300, 1, 256)

attn = torch.nn.MultiheadAttention(256, 8)

out = attn(query=query+query_pos, key=key+query_pos, value=value)

if (query is key or torch.equal(query, key) and key is value or torch.equal(key, value)):
    print('1')
print(type(str(query.device)))