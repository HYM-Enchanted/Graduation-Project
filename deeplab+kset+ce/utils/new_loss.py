import torch
import torch.nn as nn
import copy
# from .dice_loss_metrics import *

def diceCoeff(pred, gt):
    eps = 1e-5

    N = gt.size(0)  # N 是batch size
    pred = pred.view(N, -1)  # 后面512*512拉成一行

    intersection = (pred * gt).sum(1)
    unionset = pred.sum(1) + gt.sum(1)
    loss = (2 * intersection + eps) / (unionset + eps)
    return loss.sum() / N   # batch size 求平均

class SoftDiceLoss(object):
    __name__ = 'dice_loss'

    def __init__(self,num_classes):
        super(SoftDiceLoss, self).__init__()

        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(1, self.num_classes):
            temp_pred = copy.deepcopy(y_pred)
            temp_pred[temp_pred != i] = 0
            temp_target = copy.deepcopy(y_true)
            temp_target[temp_target != i] = 0

            # class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
            class_dice.append(diceCoeff(y_pred, y_true))

        mean_dice = sum(class_dice) / len(class_dice)   # 4个类别的dice求平均
        return 1 - mean_dice

