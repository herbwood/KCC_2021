import torch
import torch.nn as nn
import torch.nn.functional as F

input1 = torch.randn((8, 4))
input2 = torch.randn((8, 1))

output = torch.cat((input1, input2), dim=1)
print(output.shape)