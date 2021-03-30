import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.fpn_roi_target import fpn_roi_target
from det_oprs.loss_opr import emd_loss_softmax
from det_oprs.utils import get_padded_tensor


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()

        assert config.num_classes == 2, 'Only support two class(1fg/1bg).'

    def forward(self, image, im_info, gt_boxes=None):
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)

        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):

        loss_dict = {}
        fpn_fms = self.FPN(image)

        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes) # proposals to pool, loss 
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(rpn_rois, im_info, gt_boxes, top_k=2)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois, rcnn_labels, rcnn_bbox_targets)

        loss_dict.update(loss_dict_rpn) # loss_rpn_cls, loss_rpn_loc
        loss_dict.update(loss_dict_rcnn)

        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.cpu().detach()


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1044, 1024)

        # weight initialization 
        for l in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        # box predictor
        self.emd_pred_cls_0 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_0 = nn.Linear(1024, config.num_classes * 4)

        self.emd_pred_cls_1 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_1 = nn.Linear(1024, config.num_classes * 4)

        self.ref_pred_cls_0 = nn.Linear(1024, config.num_classes)
        self.ref_pred_delta_0 = nn.Linear(1024, config.num_classes * 4)

        self.ref_pred_cls_1 = nn.Linear(1024, config.num_classes)
        self.ref_pred_delta_1 = nn.Linear(1024, config.num_classes * 4)


        # weight initialization 
        for l in [self.emd_pred_cls_0, self.emd_pred_cls_1, self.ref_pred_cls_0, self.ref_pred_cls_1]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

        for l in [self.emd_pred_delta_0, self.emd_pred_delta_1, self.ref_pred_delta_0, self.ref_pred_delta_1]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    # rcnn_rois shape : [-1, 5]
    # rcnn_labels shape : [-1, 2]
    # rcnn_bbox_targets : [-1, 8]
    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        """
        torch.Size([512, 5]) torch.Size([512, 2]) torch.Size([512, 8])

        Pool features shape : torch.Size([512, 256, 7, 7])
        Flattened feature : torch.Size([512, 12544]) => torch.Size([512, 1024])

        # without Refinement module 
        *Box A score feature map : torch.Size([512, 2])
        *Box A loc feature map : torch.Size([512, 8])
        *Box B score feature map : torch.Size([512, 2])
        *Box B loc feature map : torch.Size([512, 8])

        Box A feature : torch.Size([512, 20])
        Box B feature : torch.Size([512, 20])

        Concated Box A feature : torch.Size([512, 1044])
        Concated Box B feature : torch.Size([512, 1044])

        Refined Box A feature : torch.Size([512, 1024])
        Refined Box B feature : torch.Size([512, 1024])

        # with Refinement module
        *Refined Box A cls feature map : torch.Size([512, 2])
        *Refined Box A loc feature map : torch.Size([512, 8])
        *Refined Box B cls feature map : torch.Size([512, 2])
        *Refined Box B loc feature map : torch.Size([512, 8])
        """

        # stride: 64,32,16,8,4 -> 4, 8, 16, 32
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")

        # print("Pool features shape :", pool_features.shape)


        # FC layers
        flatten_feature = torch.flatten(pool_features, start_dim=1) # 256 x 7 x 7

        # print("Flattened feature :", flatten_feature.shape)

        flatten_feature = F.relu_(self.fc1(flatten_feature))
        flatten_feature = F.relu_(self.fc2(flatten_feature))


        # box A
        pred_emd_cls_0 = self.emd_pred_cls_0(flatten_feature)
        pred_emd_delta_0 = self.emd_pred_delta_0(flatten_feature)

        # box B
        pred_emd_cls_1 = self.emd_pred_cls_1(flatten_feature)
        pred_emd_delta_1 = self.emd_pred_delta_1(flatten_feature)

        # box A, box B scores
        pred_emd_scores_0 = F.softmax(pred_emd_cls_0, dim=-1)
        pred_emd_scores_1 = F.softmax(pred_emd_cls_1, dim=-1)

        # print("Box A score feature map :", pred_emd_cls_0.shape)
        # print("Box A loc feature map :", pred_emd_delta_0.shape)

        # print("Box B score feature map :", pred_emd_cls_1.shape)
        # print("Box B loc feature map :", pred_emd_delta_1.shape)


        ##### Refinement module

        # concat scores with bbox loc
        boxes_feature_0 = torch.cat((pred_emd_delta_0[:, 4:], pred_emd_scores_0[:, 1][:, None]), dim=1).repeat(1, 4)
        boxes_feature_1 = torch.cat((pred_emd_delta_1[:, 4:], pred_emd_scores_1[:, 1][:, None]), dim=1).repeat(1, 4)

        # print("Box A feature :", boxes_feature_0.shape)
        # print("Box B feature :", boxes_feature_1.shape)

        # concat box A, box B with fc layer output feature map
        boxes_feature_0 = torch.cat((flatten_feature, boxes_feature_0), dim=1)
        boxes_feature_1 = torch.cat((flatten_feature, boxes_feature_1), dim=1)

        # print("Concated Box A feature :", boxes_feature_0.shape)
        # print("Concated Box B feature :", boxes_feature_1.shape)

        # relu box A, box B
        refine_feature_0 = F.relu_(self.fc3(boxes_feature_0))
        refine_feature_1 = F.relu_(self.fc3(boxes_feature_1))

        # print('Refined Box A feature :', refine_feature_0.shape)
        # print('Refined Box A feature :', refine_feature_0.shape)

        # refine box A
        pred_ref_cls_0 = self.ref_pred_cls_0(refine_feature_0)
        pred_ref_delta_0 = self.ref_pred_delta_0(refine_feature_0)

        # print("Refined Box A cls feature map :", pred_ref_cls_0.shape)
        # print("Refined Box A loc feature map :", pred_ref_delta_0.shape)

        # refine box B
        pred_ref_cls_1 = self.ref_pred_cls_1(refine_feature_1)
        pred_ref_delta_1 = self.ref_pred_delta_1(refine_feature_1)

        # print("Refined Box B cls feature map :", pred_ref_cls_1.shape)
        # print("Refined Box B loc feature map :", pred_ref_delta_1.shape)


        if self.training:

            loss0 = emd_loss_softmax(
                        pred_emd_delta_0, pred_emd_cls_0,
                        pred_emd_delta_1, pred_emd_cls_1,
                        bbox_targets, labels)

            loss1 = emd_loss_softmax(
                        pred_emd_delta_1, pred_emd_cls_1,
                        pred_emd_delta_0, pred_emd_cls_0,
                        bbox_targets, labels)

            loss2 = emd_loss_softmax(
                        pred_ref_delta_0, pred_ref_cls_0,
                        pred_ref_delta_1, pred_ref_cls_1,
                        bbox_targets, labels)

            loss3 = emd_loss_softmax(
                        pred_ref_delta_1, pred_ref_cls_1,
                        pred_ref_delta_0, pred_ref_cls_0,
                        bbox_targets, labels)

            loss_rcnn = torch.cat([loss0, loss1], axis=1) # shape : [-1, 2]
            loss_ref = torch.cat([loss2, loss3], axis=1) # shape : [-1, 2]

            # requires_grad = False
            _, min_indices_rcnn = loss_rcnn.min(axis=1)
            _, min_indices_ref = loss_ref.min(axis=1)

            loss_rcnn = loss_rcnn[torch.arange(loss_rcnn.shape[0]), min_indices_rcnn]
            loss_rcnn = loss_rcnn.mean()

            loss_ref = loss_ref[torch.arange(loss_ref.shape[0]), min_indices_ref]
            loss_ref = loss_ref.mean()

            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = loss_rcnn
            loss_dict['loss_ref_emd'] = loss_ref

            return loss_dict

        else:
            class_num = pred_ref_cls_0.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_ref_cls_0)+1
            tag = tag.repeat(pred_ref_cls_0.shape[0], 1).reshape(-1,1)

            pred_scores_0 = F.softmax(pred_ref_cls_0, dim=-1)[:, 1:].reshape(-1, 1)
            pred_scores_1 = F.softmax(pred_ref_cls_1, dim=-1)[:, 1:].reshape(-1, 1)

            pred_delta_0 = pred_ref_delta_0[:, 4:].reshape(-1, 4)
            pred_delta_1 = pred_ref_delta_1[:, 4:].reshape(-1, 4)

            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)

            pred_bbox_0 = restore_bbox(base_rois, pred_delta_0, True)
            pred_bbox_1 = restore_bbox(base_rois, pred_delta_1, True)

            pred_bbox_0 = torch.cat([pred_bbox_0, pred_scores_0, tag], axis=1)
            pred_bbox_1 = torch.cat([pred_bbox_1, pred_scores_1, tag], axis=1)

            pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), axis=1)

            return pred_bbox


def restore_bbox(rois, deltas, unnormalize=True):

    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
        
    pred_bbox = bbox_transform_inv_opr(rois, deltas)

    return pred_bbox