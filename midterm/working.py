import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_level_refine(results):
    
    outputs = []

    for i, feature_map in enumerate(results):
        target_feature_map = feature_map.clone()
        target_h, target_w = target_feature_map.shape[2:4]
        assert target_h == target_w

        for j, others in enumerate(results):
            scale = target_h // others.shape[2]

            if scale == 1:
                continue

            elif scale > 1:
                target_feature_map += F.interpolate(others, scale_factor=scale, mode='bilinear', align_corners=True)

            else:
                target_feature_map += F.max_pool2d(others, int(scale ** (-1)))

        outputs.append(target_feature_map)

    return outputs

testList = []

for i in range(4, 9):
    num = 2**i
    input = torch.randn((1, 3, num, num))
    testList.append(input)

outputs = feature_level_refine(testList)

# for i, output in enumerate(outputs):
#     print(output.eq(testList[i]))

scale = 1/2
print(int(scale ** (-1)))