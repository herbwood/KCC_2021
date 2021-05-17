import cv2
import json
import os
import numpy as np

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

if __name__ == "__main__":
    annotation_path = r"C:\Users\June hyoung Kwon\PROJECTS\KCC_2021\lib\data\annotation_val.odgt"
    image_dir_path = r"C:\Users\June hyoung Kwon\PROJECTS\images"

    records = load_json_lines(annotation_path)

    for record in records:
        id = record["ID"]
        gtboxes = record["gtboxes"]

        image_path = os.path.join(image_dir_path, id + ".jpg")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        gtlist = []

        for gtbox in gtboxes:
            gtlist.append(gtbox["fbox"])
        gtlist = np.array(gtlist)
        gtlist = xywh_to_xyxy(gtlist)

        for gtbox in gtlist:
            x1, y1, x2, y2 = gtbox.astype(int)
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        filepath = f"{id}.jpg"

        cv2.imwrite(filepath, image)

        