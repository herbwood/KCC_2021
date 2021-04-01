import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, 'lib')

from config import config
from det_oprs.anchor_generator import AnchorGenerator
from det_oprs.find_top_rpn_proposals import find_top_rpn_proposals
from det_oprs.fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss

import torch
import torch.nn.functional as F

from config import config
from det_oprs.bbox_opr import bbox_transform_inv_opr, clip_boxes_opr, filter_boxes_opr
from torchvision.ops import nms


@torch.no_grad()
def density_find_top_rpn_proposals(is_train : bool, 
                           rpn_bbox_offsets_list, 
                           rpn_cls_prob_list,
                           rpn_density_list,
                           all_anchors_list, 
                           im_info):

    # hyperparameters
    prev_nms_top_n = config.train_prev_nms_top_n if is_train else config.test_prev_nms_top_n
    post_nms_top_n = config.train_post_nms_top_n if is_train else config.test_post_nms_top_n

    batch_per_gpu = config.train_batch_per_gpu if is_train else 1
    nms_threshold = config.rpn_nms_threshold
    box_min_size = config.rpn_min_box_size

    bbox_normalize_targets = config.rpn_bbox_normalize_targets
    bbox_normalize_means = config.bbox_normalize_means
    bbox_normalize_stds = config.bbox_normalize_stds

    list_size = len(rpn_bbox_offsets_list)

    return_rois = []
    return_inds = []

    for bid in range(batch_per_gpu):

        batch_proposals_list = []
        batch_probs_list = []
        batch_density_list = []

        for l in range(list_size):


            # get proposals and probs
            # offsets ; [-1, 4]
            offsets = rpn_bbox_offsets_list[l][bid].permute(1, 2, 0).reshape(-1, 4)
            
            if bbox_normalize_targets:
                std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
                mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
                pred_offsets = pred_offsets * std_opr
                pred_offsets = pred_offsets + mean_opr

            all_anchors = all_anchors_list[l] # shape : [-1, 4]

            # real coordinates
            proposals = bbox_transform_inv_opr(all_anchors, offsets)

            if config.anchor_within_border:
                proposals = clip_boxes_opr(proposals, im_info[bid, :])


            # probs shape : [-1, 2]
            probs = rpn_cls_prob_list[l][bid].permute(1,2,0).reshape(-1, 2)
            probs = torch.softmax(probs, dim=-1)[:, 1]

            density = rpn_density_list[l][bid].permute(1,2,0).reshape(-1, 1)
            density = torch.sigmoid(density)


            # gather the proposals and probs
            batch_proposals_list.append(proposals)
            batch_probs_list.append(probs)
            batch_density_list.append(density)

        batch_proposals = torch.cat(batch_proposals_list, dim=0) # shape : [-1, 4]
        batch_probs = torch.cat(batch_probs_list, dim=0) # shape : [-1, 1]
        batch_density = torch.cat(batch_density_list, dim=0)


        # filter the zero boxes.
        batch_keep_mask = filter_boxes_opr(batch_proposals, box_min_size * im_info[bid, 2])
        batch_proposals = batch_proposals[batch_keep_mask]
        batch_probs = batch_probs[batch_keep_mask]
        batch_density = batch_density[batch_keep_mask]

        # prev_nms_top_n(prob)
        num_proposals = min(prev_nms_top_n, batch_probs.shape[0])
        batch_probs, idx = batch_probs.sort(descending=True)
        batch_probs = batch_probs[:num_proposals]

        # prev_nms_top_n(proposals)
        topk_idx = idx[:num_proposals].flatten()
        batch_proposals = batch_proposals[topk_idx] # shape : [12000, 4]

        # prev_nms_top_n(density)
        batch_density = batch_density[topk_idx]
        
        # For each image, run a total-level NMS, and choose topk results.
        keep = nms(batch_proposals, batch_probs, nms_threshold)
        keep = keep[:post_nms_top_n]
        batch_proposals = batch_proposals[keep] # shape [2000, 4]

        batch_density = batch_density[keep]

        #batch_probs = batch_probs[keep]
        # cons the rois
        batch_inds = torch.ones(batch_proposals.shape[0], 1).type_as(batch_proposals) * bid
        batch_rois = torch.cat([batch_inds, batch_proposals], axis=1) # shape : [2000, 5] (bid(), x, y , w, h)
        return_rois.append(batch_rois)

    if batch_per_gpu == 1:
        return batch_rois, batch_density

    else:
        concated_rois = torch.cat(return_rois, axis=0)
        return concated_rois, batch_density



import torch
import numpy as np

from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr
from config import config


def fpn_rpn_reshape(pred_cls_score_list, pred_bbox_offsets_list):

    final_pred_bbox_offsets_list = []
    final_pred_cls_score_list = []

    for bid in range(config.train_batch_per_gpu):

        batch_pred_bbox_offsets_list = []
        batch_pred_cls_score_list = []

        for i in range(len(pred_cls_score_list)):
            pred_cls_score_perlvl = pred_cls_score_list[i][bid].permute(1, 2, 0).reshape(-1, 2)
            pred_bbox_offsets_perlvl = pred_bbox_offsets_list[i][bid].permute(1, 2, 0).reshape(-1, 4)
            batch_pred_cls_score_list.append(pred_cls_score_perlvl)
            batch_pred_bbox_offsets_list.append(pred_bbox_offsets_perlvl)

        batch_pred_cls_score = torch.cat(batch_pred_cls_score_list, dim=0)
        batch_pred_bbox_offsets = torch.cat(batch_pred_bbox_offsets_list, dim=0)

        final_pred_cls_score_list.append(batch_pred_cls_score)
        final_pred_bbox_offsets_list.append(batch_pred_bbox_offsets)

    final_pred_cls_score = torch.cat(final_pred_cls_score_list, dim=0)
    final_pred_bbox_offsets = torch.cat(final_pred_bbox_offsets_list, dim=0)

    # final_pred-cls_score shape : [-1, 2]
    # final_pred-bbox-offsets shape : [-1, 4]
    return final_pred_cls_score, final_pred_bbox_offsets


def density_fpn_anchor_target_opr_core_impl(gt_boxes, im_info, anchors, allow_low_quality_matches=True):
    """
    returns labels whether pos/neg/ignore and convert bbox target as regression form
    """
    ignore_label = config.ignore_label # -1

    # get the gt boxes
    # in_info[5] : number of gt boxes
    valid_gt_boxes = gt_boxes[:int(im_info[5]), :] # get all the gtboxes
    valid_gt_boxes = valid_gt_boxes[valid_gt_boxes[:, -1].gt(0)] # get gtboxes if it exsists


    # compute the iou matrix
    # returns (number of anchors) x (number of valid gtboxes) iou values
    anchors = anchors.type_as(valid_gt_boxes)
    overlaps = box_overlap_opr(anchors, valid_gt_boxes[:, :4])

    # density of ground truth boxes
    # shape : [number of gt boxes, number of gt boxes] iou values
    density_overlaps = box_overlap_opr(valid_gt_boxes[:, :4], valid_gt_boxes[:, :4]) 

    # match the dtboxes
    # return max values by columns : highest iou 
    # max_overlaps shape : [#anchors, 1]
    max_overlaps, argmax_overlaps = torch.max(overlaps, axis=1)

    density_overlaps, density_argmax_overlaps = torch.topk(density_overlaps, dim=1, k=2)
    density_overlaps = density_overlaps[:, -1].unsqueeze(-1)
    density_argmax_overlaps = density_argmax_overlaps[:, -1] # [number of ground truth boxes]

    density_map = torch.cat((valid_gt_boxes[:, :4], density_overlaps), dim=1)

    #_, gt_argmax_overlaps = torch.max(overlaps, axis=0)
    gt_argmax_overlaps = my_gt_argmax(overlaps)

    del overlaps

    # all ignore
    labels = torch.ones(anchors.shape[0], device=gt_boxes.device, dtype=torch.long) * ignore_label

    # set negative ones
    # negative label = -1
    labels = labels * (max_overlaps >= config.rpn_negative_overlap)

    # set positive ones
    fg_mask = (max_overlaps >= config.rpn_positive_overlap)

    if allow_low_quality_matches:
        gt_id = torch.arange(valid_gt_boxes.shape[0]).type_as(argmax_overlaps)
        argmax_overlaps[gt_argmax_overlaps] = gt_id
        max_overlaps[gt_argmax_overlaps] = 1
        fg_mask = (max_overlaps >= config.rpn_positive_overlap)

    # set positive ones
    # labels : info about positive/negative anchors
    fg_mask_ind = torch.nonzero(fg_mask, as_tuple=False).flatten()
    labels[fg_mask_ind] = 1

    # bbox targets
    # Transform the bounding box and ground truth to the loss targets
    # shape : [-1, 4]
    print(valid_gt_boxes[argmax_overlaps, :4].shape)
    bbox_targets =  bbox_transform_opr(anchors, valid_gt_boxes[argmax_overlaps, :4])

    if config.rpn_bbox_normalize_targets:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
        minus_opr = mean_opr / std_opr
        bbox_targets = bbox_targets / std_opr - minus_opr

    # labels : positive = 1, negative = 0, ignore = -1
    # bbox_targets shape : [-1, 4]
    # print(anchors.shape, labels.shape, bbox_targets.shape, density_targets.shape)
    return labels, bbox_targets


@torch.no_grad()
def density_fpn_anchor_target(boxes, im_info, all_anchors_list):

    final_labels_list = []
    final_bbox_targets_list = []
    final_density_targets_list = []

    for bid in range(config.train_batch_per_gpu):

        batch_labels_list = []
        batch_bbox_targets_list = []
        batch_density_targets_list = []

        for i in range(len(all_anchors_list)):
            anchors_perlvl = all_anchors_list[i]  # shape : [-1, 4]
            rpn_labels_perlvl, rpn_bbox_targets_perlvl = density_fpn_anchor_target_opr_core_impl(
                boxes[bid], im_info[bid], anchors_perlvl)

            batch_labels_list.append(rpn_labels_perlvl)
            batch_bbox_targets_list.append(rpn_bbox_targets_perlvl)

        # here we samples the rpn_labels
        concated_batch_labels = torch.cat(batch_labels_list, dim=0) # shape : [-1, 1]
        concated_batch_bbox_targets = torch.cat(batch_bbox_targets_list, dim=0) # shape : [-1, 4]

        # sample labels
        pos_idx, neg_idx = subsample_labels(concated_batch_labels,
            config.num_sample_anchors, config.positive_anchor_ratio) # 0.5
        concated_batch_labels.fill_(-1)
        concated_batch_labels[pos_idx] = 1
        concated_batch_labels[neg_idx] = 0

        final_labels_list.append(concated_batch_labels)
        final_bbox_targets_list.append(concated_batch_bbox_targets)

    final_labels = torch.cat(final_labels_list, dim=0)
    final_bbox_targets = torch.cat(final_bbox_targets_list, dim=0)
    
    # final_labels : [-1, 1]
    # final-bbox_targets : [-1, 4]
    output = torch.unique(final_bbox_targets)
    return final_labels, final_bbox_targets


def my_gt_argmax(overlaps):

    gt_max_overlaps, _ = torch.max(overlaps, axis=0)
    gt_max_mask = overlaps == gt_max_overlaps
    gt_argmax_overlaps = []

    for i in range(overlaps.shape[-1]):
        gt_max_inds = torch.nonzero(gt_max_mask[:, i], as_tuple=False).flatten()
        gt_max_ind = gt_max_inds[torch.randperm(gt_max_inds.numel(), device=gt_max_inds.device)[0,None]]
        gt_argmax_overlaps.append(gt_max_ind)

    gt_argmax_overlaps = torch.cat(gt_argmax_overlaps)

    return gt_argmax_overlaps


def subsample_labels(labels, 
                    num_samples : int, 
                    positive_fraction : float):
    """returns random index of positive/negative labels"""

    positive = torch.nonzero((labels != config.ignore_label) & (labels != 0), as_tuple=False).squeeze(1)
    negative = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    
    return pos_idx, neg_idx


class DensityRPN(nn.Module):
    def __init__(self, rpn_channel = 256):

        super().__init__()

        self.training = True

        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)

        self.rpn_conv = nn.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2d(rpn_channel, config.num_cell_anchors * 2, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = nn.Conv2d(rpn_channel, config.num_cell_anchors * 4, kernel_size=1, stride=1)


        ####################Density############################################
        self.rpn_density = nn.Conv2d(rpn_channel, 128, kernel_size=1, stride=1)
        self.density_conv = nn.Conv2d(128 + config.num_cell_anchors * 2 + config.num_cell_anchors * 4, 
                                        3, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features, im_info, boxes=None):

        # prediction
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        density_feature_map_list = []
        pred_density_list = []

        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_cls_score(t))
            pred_bbox_offsets_list.append(self.rpn_bbox_offsets(t))
            density_feature_map_list.append(self.rpn_density(t))

        for cls, bbox, density in zip(pred_cls_score_list, pred_bbox_offsets_list, density_feature_map_list):
            concated_feature_map = torch.cat((cls, bbox, density), 1)
            density_conv_feature_map = self.density_conv(concated_feature_map)
            pred_density_list.append(density_conv_feature_map)

        # get anchors
        all_anchors_list = []

        # stride: 64,32,16,8,4 p6->p2
        base_stride = 4
        off_stride = 2**(len(features)-1) # 16

        for fm in features:
            layer_anchors = self.anchors_generator(fm, base_stride, off_stride)
            off_stride = off_stride // 2
            all_anchors_list.append(layer_anchors)

        # sample from the predictions
        # rpn_rois shape : [2000, 5]
        rpn_rois, batch_density = density_find_top_rpn_proposals(
                self.training, pred_bbox_offsets_list, pred_cls_score_list, pred_density_list,
                all_anchors_list, im_info)
        rpn_rois = rpn_rois.type_as(features[0])


        if self.training:
            # rpn_labels shape : [-1, 1], pos/neg/ignore
            # rpn_bbox_targets : [-1, 4] bbox target coords
            rpn_labels, rpn_bbox_targets = density_fpn_anchor_target(
                    boxes, im_info, all_anchors_list)

            #rpn_labels = rpn_labels.astype(np.int32)
            # pred_cls_score shape : [-1, 2]
            # pred_bbox_offsets shape : [-1, 4]
            pred_cls_score, pred_bbox_offsets = fpn_rpn_reshape(
                pred_cls_score_list, pred_bbox_offsets_list)


            # rpn loss

            # objectness loss
            # only consider positive/negative anchors
            valid_masks = rpn_labels >= 0
            objectness_loss = softmax_loss(
                pred_cls_score[valid_masks],
                rpn_labels[valid_masks])

            # ignore other anchors
            pos_masks = rpn_labels > 0
            localization_loss = smooth_l1_loss(
                pred_bbox_offsets[pos_masks],
                rpn_bbox_targets[pos_masks],
                config.rpn_smooth_l1_beta)

            normalizer = 1 / valid_masks.sum().item()
            loss_rpn_cls = objectness_loss.sum() * normalizer
            loss_rpn_loc = localization_loss.sum() * normalizer

            loss_dict = {}
            loss_dict['loss_rpn_cls'] = loss_rpn_cls
            loss_dict['loss_rpn_loc'] = loss_rpn_loc
            
            return rpn_rois, loss_dict
        else:
            return rpn_rois