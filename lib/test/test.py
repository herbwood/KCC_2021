import sys
sys.path.insert(0, 'lib')

from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from data.CrowdHuman import CrowdHuman
from det_oprs import *
from layers import batch_norm, pooler
from module.rpn import RPN
from utils import misc_utils, nms_utils
from config import config
from det_oprs.utils import get_padded_tensor

import torch


if __name__ == "__main__":

    resnet50 = ResNet50(2, False)
    FPN = FPN(resnet50, 2, 6)
    RPN = RPN(rpn_channel=256)

    crowdhuman = CrowdHuman(config, if_train=True)
    data_iter = torch.utils.data.DataLoader(dataset=crowdhuman,
                batch_size=1,
                num_workers=2,
                collate_fn=crowdhuman.merge_batch,
                shuffle=True)

    for (images, gt_boxes, im_info) in data_iter:

        images = get_padded_tensor(images, 64)
        fpn_fms = FPN(images)

        # # for output in fpn_fms:
        # #     print(output.shape)
        rpn_rois, loss_dict_rpn = RPN(fpn_fms, im_info, gt_boxes)
        # print(rpn_rois.shape)
        print(loss_dict_rpn)



