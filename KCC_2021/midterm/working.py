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

path = r"C:\Users\June hyoung Kwon\PROJECTS\images"
annotation_path = r"C:\Users\June hyoung Kwon\PROJECTS\KCC_2021\lib\data\annotation_train.odgt"

records = load_json_lines(annotation_path)
coordlist = []
passcoordlist = [(611, 864, 639, 932), (656, 867, 674, 922), (714, 872, 737, 929), (758, 890, 787, 992),
                 (40, 1251, 401, 1368), (195, 1422, 680, 1535), (543, 1235, 908, 1293), (816, 1349, 1104, 1419),
                 (1033, 1153, 1119, 1190), (1287, 1266, 1425, 1318), (1061, 904, 1073, 929)]

# for i in range(len(records)):
#     coordlist = []
#     id = records[i]["ID"]
#     filename = records[i]["ID"] + ".jpg"
#     if filename == "273271,2f58400023fe3e6f.jpg":
#         image_path = os.path.join(path, filename)

#         for gtbox in records[i]["gtboxes"]:
#             coordlist.append(gtbox["fbox"])

#         coordlist = torch.tensor(coordlist, dtype=torch.float)
#         coordlist = xywh_to_xyxy(coordlist)


#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)

#         for i in range(len(coordlist)):
#             one_box = coordlist[i]
#             (x1, y1, x2, y2) = np.array(one_box[:4]).astype(int)

#             if (x1, y1, x2, y2) in passcoordlist:
#                 continue

#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

#             cv2.imwrite(f"{i}_{x1}_{y1}_{x2}_{y2}.jpg", image)
#         cv2.imwrite(f"0000000000000000{id}.jpg", image)


# 1 : 2, 4, 6
fplist1 = [(1069, 899, 1111, 1017), (598, 918, 782, 1290), (221, 873, 459, 1321), (1006, 912, 1096, 1102), (251, 873, 344, 1090)]
missedlist1 = [(496, 856, 519, 920), (662, 864, 684, 921), (766, 865, 791, 924)]
# 2 : 1, 2, 4
fplist2 = [(224, 865, 438, 1308), (880, 909, 1074, 1287), (598, 918, 782, 1290)]
missedlist2 = [(496, 856, 519, 920), (662, 864, 684, 921), (477, 853, 499, 921)]

image_path1 = r"C:\Users\June hyoung Kwon\PROJECTS\KCC_2021\tools\retina_fpn_baseline_03\30_750_877_786_963.jpg"
image_path2 = r"C:\Users\June hyoung Kwon\PROJECTS\KCC_2021\tools\retina_emd_diou_04\25_669_899_707_1000.jpg"

image = image = cv2.imread(image_path2, cv2.IMREAD_COLOR)

for i in range(len(fplist2)):
    one_box = fplist2[i]
    (x1, y1, x2, y2) = np.array(one_box[:4]).astype(int)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

for i in range(len(missedlist1)):
    one_box = missedlist2[i]
    (x1, y1, x2, y2) = np.array(one_box[:4]).astype(int)

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)

cv2.imwrite(f"red_yellow_boxed2.jpg", image)