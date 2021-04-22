import json
import os
import torch

from numpy.lib.npyio import load

def load_json_lines(fpath : str):
    
    assert os.path.exists(fpath)

    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]

    return records

def box_overlap_opr(box, gt):
    """ Find IoU(Intersection over Union) between 
    bounding box and ground truth box"""

    # N : number of anchor boxes
    # m: number of ground truth boxes
    assert box.ndim == 2
    assert gt.ndim == 2

    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)

    # box[:, None, 2:].shape : [-1, 1, 2]
    # gt[:, :2].shape : [-1, 2]
    width_height = torch.min(box[:, None, 2:], gt[:, 2:]) - torch.max(box[:, None, :2], gt[:, :2]) + 1  # [N,M,2]
    width, height = width_height[:, :, 0], width_height[:, :, 1]
    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]

    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area_box[:, None] + area_gt - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )

    return width, height, iou

def xywh_to_xyxy(boxes):
    
    assert boxes.shape[1]>=4
    boxes[:, 2:4] += boxes[:,:2]

    return boxes

# test_annotation_path = r"midterm\test_data\test_annotation.odgt"
# records = load_json_lines(test_annotation_path)

# aspect_ratio_list = []

# for i in range(len(records)):
#     coordlist = []
#     for gtbox in records[i]["gtboxes"]:
#         coordlist.append(gtbox["fbox"])
#     coordlist = torch.tensor(coordlist, dtype=torch.float)
#     coordlist = xywh_to_xyxy(coordlist)
    # print(records[i]["ID"], len(records[i]["gtboxes"]))
    # width, height, iou = box_overlap_opr(coordlist, coordlist)
    # print(iou)

    # aspect_ratio = width / height
    # aspect_ratio_list.append(aspect_ratio)

# print(aspect_ratio)

# train_annotation_path = r"lib\data\annotation_train.odgt"
# validation_annotation_path = r"lib\data\annotation_val.odgt"

# train_id_list = []
# validation_id_list = []

# train_records = load_json_lines(train_annotation_path)

# for i in range(len(train_records[:10])):
#     train_id_list.append(train_records[i]["ID"])
# print(train_id_list)

output = torch.exp(torch.tensor(1, 2), 1)
print(output)

