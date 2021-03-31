import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.randn((1, 3, 64, 64))
downsample = nn.MaxPool2d(kernel_size=2)
output = downsample(input)

print(output.shape)