import sys

sys.path.insert(0, 'lib')
sys.path.insert(0, 'model')

from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from data.CrowdHuman import CrowdHuman
from det_oprs import *
from layers import batch_norm, pooler
from module.rpn import RPN
from utils import misc_utils, nms_utils
from config import config
from det_oprs.utils import get_padded_tensor
from det_oprs.fpn_roi_target import fpn_roi_target

from rcnn_emd_double_refine.network import RCNN
from layers.pooler import roi_pooler

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence

from module.density_rpn import DensityRPN, density_fpn_roi_target
from rcnn_emd_density_refine.network import RCNN


if __name__ == "__main__":

    resnet50 = ResNet50(2, False)
    FPN = FPN(resnet50, 2, 6)
    RPN = RPN(rpn_channel=256)
    DensityRPN = DensityRPN(rpn_channel=256)
    RCNN = RCNN()

    crowdhuman = CrowdHuman(config, if_train=True)
    data_iter = torch.utils.data.DataLoader(dataset=crowdhuman,
                batch_size=1,
                num_workers=2,
                collate_fn=crowdhuman.merge_batch,
                shuffle=True)

    loss_dict = {}

    for (images, gt_boxes, im_info) in data_iter:

        images = get_padded_tensor(images, 64)
        fpn_fms = FPN(images)

        # for output in fpn_fms:
        #     """
        #     Output shape example :
        #     torch.Size([1, 256, 13, 19])
        #     torch.Size([1, 256, 26, 38])
        #     torch.Size([1, 256, 52, 76])
        #     torch.Size([1, 256, 104, 152])
        #     torch.Size([1, 256, 208, 304])
        #     """
        #     print(output.shape)

        rpn_rois, loss_dict_rpn = DensityRPN(fpn_fms, im_info, gt_boxes)
        # rcnn_rois shape : [-1, 5]
        # rcnn_labels shape : [-1, 2]
        # rcnn_bbox_targets : [-1, 8]
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = density_fpn_roi_target(rpn_rois, im_info, gt_boxes, top_k=2)
        # print(rcnn_rois.shape, rcnn_labels.shape, rcnn_bbox_targets.shape)

        # fpn_fms = fpn_fms[1:][::-1]
        # stride = [4, 8, 16, 32]
        # pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")

        # loss_dict_rcnn = RCNN(fpn_fms, rcnn_rois, rcnn_labels, rcnn_bbox_targets)

        # loss_dict.update(loss_dict_rpn) # loss_rpn_cls, loss_rpn_loc
        # loss_dict.update(loss_dict_rcnn)