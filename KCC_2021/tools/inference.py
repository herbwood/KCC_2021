import os
import sys
import argparse

import cv2
import torch
import numpy as np

sys.path.insert(0, '../lib')

from utils import misc_utils, visual_utils, nms_utils


import time


def inference(args, config, network, dir=False):

    passcoordlist = [(611, 864, 639, 932), (656, 867, 674, 922), (714, 872, 737, 929), (758, 890, 787, 992),
                 (40, 1251, 401, 1368), (195, 1422, 680, 1535), (543, 1235, 908, 1293), (816, 1349, 1104, 1419),
                 (1033, 1153, 1119, 1190), (1287, 1266, 1425, 1318), (1061, 904, 1073, 929)]

    # model_path
    misc_utils.ensure_dir('outputs')
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    model_file = os.path.join(saveDir,
                            'dump-{}.pth'.format(args.resume_weights))
    assert os.path.exists(model_file)

    # build network
    net = network()
    net.eval()
    check_point = torch.load(model_file, map_location=torch.device('cpu'))
    net.load_state_dict(check_point['state_dict'])

    

    if 1==0:
        # get data
        for image_file in os.listdir(args.img_path):
            image, resized_img, im_info = get_data(os.path.join(args.img_path, image_file), #args.img_path, 
                                                    config.eval_image_short_size, 
                                                    config.eval_image_max_size)
            pred_boxes = net(resized_img, im_info).numpy()
            pred_boxes = post_process(pred_boxes, config, im_info[0, 2], args)
            # retina emd simple : 2.625977039337158
            # retina emd repulsion diou : 2.689812660217285
            # retina emd repulsion diou refine : 2.5761098861694336
            pred_tags = pred_boxes[:, 5].astype(np.int32).flatten()
            pred_tags_name = np.array(config.class_names)[pred_tags]
            

            # inplace draw
            image = visual_utils.draw_boxes(
                    image,
                    pred_boxes[:, :4],
                    scores=pred_boxes[:, 4],
                    tags=pred_tags_name,
                    line_thick=4, line_color='blue')

            # name = args.img_path.split('/')[-1].split('.')[-2]
            # fpath = 'outputs/{}.png'.format(name)
            fpath = f"{args.output_path}/{image_file}"
            
            cv2.imwrite(fpath, image)
            print(fpath)
    else:
        image, resized_img, im_info = get_data(args.img_path, #args.img_path, 
                                            config.eval_image_short_size, 
                                            config.eval_image_max_size)
        pred_boxes = net(resized_img, im_info).numpy()
        pred_boxes = post_process(pred_boxes, config, im_info[0, 2], args)
        # retina emd simple : 2.625977039337158
        # retina emd repulsion diou : 2.689812660217285
        # retina emd repulsion diou refine : 2.5761098861694336
        pred_tags = pred_boxes[:, 5].astype(np.int32).flatten()
        pred_tags_name = np.array(config.class_names)[pred_tags]

        for i in range(pred_boxes.shape[0]):
            one_box = pred_boxes[i, :]
            (x1, y1, x2, y2) = np.array(one_box[:4]).astype(int)

            if (x1, y1, x2, y2) in passcoordlist:
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

            cv2.imwrite(f"{i}_{x1}_{y1}_{x2}_{y2}.jpg", image)
        cv2.imwrite(f"0000000000000000{id}.jpg", image)
        

    
def post_process(pred_boxes, config, scale, args):

    if config.test_nms_method == 'set_nms':
        assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"

        top_k = pred_boxes.shape[-1] // 6 # 2
        n = pred_boxes.shape[0] # number of predictions 
        pred_boxes = pred_boxes.reshape(-1, 6) # [n, (x, y, w, h, confidence score, tag)]

        idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1) # [1, 1, 2, 2, 3, 3, ..., n, n]
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, 4] > config.pred_cls_threshold 
        pred_boxes = pred_boxes[keep]

        keep = nms_utils.set_cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]

    elif config.test_nms_method == 'normal_nms':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.cpu_nms(pred_boxes, config.test_nms)
        pred_boxes = pred_boxes[keep]

    elif config.test_nms_method == 'none':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]

    pred_boxes[:, :4] /= scale
    # keep = pred_boxes[:, 4] > config.visulize_threshold
    keep = pred_boxes[:, 4] > args.vthresh
    pred_boxes = pred_boxes[keep]

    return pred_boxes


def get_data(img_path, short_size, max_size):

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized_img, scale = resize_img(image, short_size, max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    resized_img = resized_img.transpose(2, 0, 1)
    im_info = np.array([height, width, scale, original_height, original_width, 0])

    return image, torch.tensor([resized_img]).float(), torch.tensor([im_info])


def resize_img(image, short_size, max_size):
    height = image.shape[0]
    width = image.shape[1]
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
        
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    resized_image = cv2.resize(
            image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale


def run_inference():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--vthresh', '-v', default=0.5, type=float)
    parser.add_argument('--img_path', '-i', default=None, required=True, type=str)
    parser.add_argument('--output_path', '-o', default=None, required=True, type=str)
    parser.add_argument('--dirorfile', '-d', default=False, type=bool)
    args = parser.parse_args()

    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)

    from config import config
    from network import Network
    
    inference(args, config, Network, dir=args.dirorfile)


if __name__ == '__main__':
    run_inference()