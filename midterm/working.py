import torch

input0 = torch.ones(10, 4)
input1 = torch.zeros(10, 4)
output = torch.cat([input0, input1], axis=1)
output1, output2 = output[:, :4], output[:, 4:]
print(output.reshape(-1, 4))
final_output = torch.cat([output2, output1], axis=1).reshape(-1, 4)
print(final_output)