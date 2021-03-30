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

from rcnn_emd_refine.network import RCNN

import torch


if __name__ == "__main__":

    resnet50 = ResNet50(2, False)
    FPN = FPN(resnet50, 2, 6)
    RPN = RPN(rpn_channel=256)
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

        # # for output in fpn_fms:
        # #     print(output.shape)
        rpn_rois, loss_dict_rpn = RPN(fpn_fms, im_info, gt_boxes)

        # rcnn_rois shape : [-1, 5]
        # rcnn_labels shape : [-1, 2]
        # rcnn_bbox_targets : [-1, 8]
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(rpn_rois, im_info, gt_boxes, top_k=2)
        print(rcnn_rois.shape, rcnn_labels.shape, rcnn_bbox_targets.shape)

        loss_dict_rcnn = RCNN(fpn_fms, rcnn_rois, rcnn_labels, rcnn_bbox_targets)

        loss_dict.update(loss_dict_rpn) # loss_rpn_cls, loss_rpn_loc
        loss_dict.update(loss_dict_rcnn)

        



