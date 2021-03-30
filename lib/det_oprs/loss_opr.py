import torch

from config import config


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