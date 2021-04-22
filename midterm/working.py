import cv2
import json
import os
import torch 
import numpy as np
import cv2

def load_json_lines(fpath : str):
    
    assert os.path.exists(fpath)

    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]

    return records

def xywh_to_xyxy(boxes):
    
    assert boxes.shape[1]>=4
    boxes[:, 2:4] += boxes[:,:2]

    return boxes

path = r"C:\Users\June hyoung Kwon\PROJECTS\KCC_2021\midterm\test_data\image"
annotation_path = r"C:\Users\June hyoung Kwon\PROJECTS\KCC_2021\midterm\test_data\test_annotation.odgt"

records = load_json_lines(annotation_path)
coordlist = []

for i in range(len(records)):
    coordlist = []
    filename = records[i]["ID"] + ".jpg"
    image_path = os.path.join(path, filename)

    for gtbox in records[i]["gtboxes"]:
        coordlist.append(gtbox["fbox"])

    coordlist = torch.tensor(coordlist, dtype=torch.float)
    coordlist = xywh_to_xyxy(coordlist)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for i in range(len(coordlist)):
        one_box = coordlist[i]
        x1, y1, x2, y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imwrite(f"test{i}.png", image)