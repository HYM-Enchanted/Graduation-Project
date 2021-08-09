import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
# from .dice_loss_metrics import *

#这个4个类别求平均的dice loss，avg dice loss

class My_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,  pred, gt):
        eps = 1e-5
        intersection = (pred * gt).sum()
        unionset = pred.sum() + gt.sum()
        dice = (2 * intersection + eps) / (unionset + eps)

        return dice



class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss

        elif mode == 'multi':  # 多个loss结合
            multi_loss = self.Multi_loss

            return multi_loss

        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss


    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


    def Multi_loss(self,logit, target):
        n, c, h, w = logit.size()

        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss_ce = criterion(logit, target.long())

        if self.batch_average:
            loss_ce /= n

        # 以上得到了wce loss，以下计算dice loss，然后结合，可以改比例，但是目前是一比一

        pred = torch.argmax(logit,dim= 1)   # 对应预测的类别
        # print('check 1:',torch.max(logit),torch.min(logit))
        log_predict = F.softmax(logit,dim=1)   # sogtmax 0-1
        # print('cccccc:',log_predict.size())
        # print('check 2:', torch.max(log_predict), torch.min(log_predict))
        pred_p = torch.max(log_predict ,dim = 1)[0]  # 每一个pixel的最大概率对应的概率，跟上面对应的类别对应
        # print("yuayudyafa:",torch.max(pred_p),torch.min(pred_p))
        self.num_class = 5
        # 多个类别分别计算dice
        class_dice = []

        criterion2 = My_Loss()
        if self.cuda:
            criterion2 = criterion2.cuda()

        for i in range(1, self.num_class):  # 第1个类别到第5个类别，就是1,2,3,4桐花树，无瓣海桑，茳芏，秋茄
            temp_target = torch.zeros(target.size())
            temp_target[target == i] = 1

            temp_pred = torch.zeros(target.size()).cuda()
            temp_pred[pred == i] = 1   # 选到对应的类别，造一个one
            # print('???',torch.unique(temp_pred))
            temp_pred *= pred_p   #该pixel预测类别对应的概率
            # print(temp_pred.size())
            # print(torch.max(temp_pred),torch.min(temp_pred))


            class_dice.append(criterion2(temp_pred.cuda(), temp_target.cuda()))
        print('class dice:',class_dice,'\n')
        mean_dice = sum(class_dice) / len(class_dice)  # 4个类别的dice求平均
        loss_dice = 1 - mean_dice   # 注意是1减去整个batch的mean_dice之后再求batch的平均
        loss_dice /= n    # 除以batch size大小，求平均每张图的dice loss，前面的wce loss也是求平均每张图的wce loss
        print('return loss:',loss_ce,'\n',loss_dice,'\n')   #还是想打印一下loss
        # loss= loss_ce + loss_dice
        return loss_ce,loss_dice   # 分别返回loss，可以画tensorboard
        # return loss_dice   # 检查只有一个dice loss有没有梯度，能不能反向传播，能！！！
        # print('return loss:',loss_ce,'\n',loss_dice)
        # return loss



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




