import torch
import torch.nn.functional as F

def feature_level_refine(results):
    
    outputs = []

    for i, feature_map in enumerate(results):
        target_feature_map = feature_map.clone()
        target_h, target_w = target_feature_map.shape[2:4]
        # assert target_h == target_w

        for j, others in enumerate(results):
            scale = target_h / others.shape[2]

            if scale == 1:
                continue

            elif scale > 1:
                target_feature_map += F.interpolate(others, scale_factor=int(scale), mode='bilinear', align_corners=True)

            else:
                target_feature_map += F.max_pool2d(others, int(scale ** (-1)))

        outputs.append(target_feature_map)

    return outputs


def feature_level_balanced_refine(results):

    outputs = []
    scales = []
    target_idx = len(results) // 2 + 1
    target_feature_map = results[target_idx].clone()
    target_h, target_w = target_feature_map.shape[2:4]

    for i, feature_map in enumerate(results):
        feature_shape = feature_map.shape[2]
        scale = target_h / feature_shape
        scales.append(scale)

        if scale == 1:
            continue

        elif scale > 1:
            scaled_feature_map = F.interpolate(feature_map, 
                                                scale_factor=int(scale), mode='bilinear', align_corners=True)

        else:
            scaled_feature_map = F.max_pool2d(feature_map, int(scale ** (-1)))

        target_feature_map += scaled_feature_map
        target_feature_map /= len(results)

    for scale in scales:

        if scale == 1:
            outputs.append(target_feature_map)

        elif scale > 1:
            final_scaled_feature_map = F.max_pool2d(target_feature_map, int(scale))
            outputs.append(final_scaled_feature_map)

        else:
            final_scaled_feature_map = F.interpolate(target_feature_map, 
                                                    scale_factor=int(scale ** (-1)), mode='bilinear', align_corners=True)
            outputs.append(final_scaled_feature_map)

    return outputs
        


if __name__ == "__main__":
    testList = []

    for i in range(4, 9):
        num = 2**i
        input = torch.randn((1, 3, num, num))
        testList.append(input)

    outputs = feature_level_balanced_refine(testList)
    print(outputs[1])

    for output in outputs:
        print(output.shape)

    
