import torch

from config import config

from det_oprs.bbox_opr import bbox_transform_inv_opr


def softmax_loss(score, label, ignore_label=-1):

    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]

    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask

    onehot = torch.zeros(vlabel.shape[0], config.num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)

    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask

    return loss


def smooth_l1_loss(pred, target, beta: float):

    if beta < 1e-5:
        loss = torch.abs(input - target)

    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)

    return loss.sum(axis=1)


def emd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    """
    Input example 

    p_b0 : Refined Box A loc feature map : torch.Size([512, 8])
    p_s0 : Refined Box A cls feature map : torch.Size([512, 2])
    p_b1 : Refined Box B loc feature map : torch.Size([512, 8])
    p_s1 : Refined Box B cls feature map : torch.Size([512, 2])
    
    targets : torch.Size([512, 8])
    labels : torch.Size([512, 2])
    """

    # reshape
    
    # pred_delta shape 
    # torch.Size([512, 8]), torch.Size([512, 8])
    # torch.Size([512, 16])
    # torch.Size([1024, 8])
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])

    # pred_score shape
    # torch.Size([512, 2]), torch.Size([512, 2])
    # torch.Size([512, 4])
    # torch.Size([1024, 2])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])

    # target shape : torch.Size([1024, 4])
    targets = targets.reshape(-1, 4)

    # labels shape : torch.Size([1024])
    labels = labels.long().flatten()

    # cons masks
    valid_masks = labels >= 0 # remove ignore labels
    fg_masks = labels > 0 # only positive labels 

    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4) # [1024, 2, 4]
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]

    # loss for regression
    # only get loss for positive samples 
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)

    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks

    # total loss
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    
    return loss.reshape(-1, 1)

def focal_loss(inputs, targets, alpha=-1, gamma=2):

    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs)
    neg_pred = inputs ** gamma * torch.log(1 - inputs)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)

    return loss.sum(axis=1)
    

def emd_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels):

    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    print("Label shape before loss part:", labels.shape)
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)

    print("Label shape(loss part) :", labels.shape)

    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels, config.focal_loss_alpha, config.focal_loss_gamma)

    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            config.smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    loss = loss.reshape(-1, 1)

    return loss

def bbox_with_delta(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr

    pred_bbox = bbox_transform_inv_opr(rois, deltas)

    return pred_bbox



def diou_loss(delta, anchors, bboxes2):
    bboxes1 = bbox_with_delta(delta, anchors)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))

    if rows * cols == 0:
        return dious

    exchange = False

    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    # pred bbox, gt box wdith and height 
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    # pred bbox, gt box area 
    area1 = w1 * h1
    area2 = w2 * h2

    # pred bbox, gt box center point coord 
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    # inter area coord
    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    # outer area coord for C
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    # intersection area
    # diagonal distance between pred box and gt box
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    # outer area 
    # diagnoal distance of outer rectangle C
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)

    # Union area of pred bbox and gt box
    union = area1 + area2 - inter_area

    # DIoU 
    # min : -1, max : 1
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0) 

    if exchange:
        dious = dious.T
    dious = dious.reshape(-1, 1)
    loss = 1.0 - dious
    loss = loss.sum(axis=1)

    return loss


def emd_loss_diou(p_b0, p_s0, p_b1, p_s1, targets, labels, anchors):

    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1]) 

    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    anchors = anchors.reshape(-1, 4)

    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels, config.focal_loss_alpha, config.focal_loss_gamma)

    fg_masks = (labels > 0).flatten()
    localization_loss = diou_loss(pred_delta[fg_masks], anchors[fg_masks], targets[fg_masks])

    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    
    return loss.reshape(-1, 1)


