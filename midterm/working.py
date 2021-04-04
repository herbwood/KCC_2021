import torch
import torch.nn as nn
import torch.nn.functional as F


torch.Size([38, 5])
torch.Size([858])
torch.Size([858, 4])

input = torch.randn((38, 5))
args = torch.randint(1, 100, 100)
output = input[args, :4]
print(output.shape)
