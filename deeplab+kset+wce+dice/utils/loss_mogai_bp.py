import torch
import torch.nn as nn
import copy
import numpy as np
import torch.nn.functional as F
# from .dice_loss_metrics import *

class My_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,  pred, gt):  # 这里计算的是一整个batch内该类别的dice loss
        eps = 1e-5
        # N = pred.shape[0] # N 是batch size

        # pred = torch.from_numpy(pred)
        # pred.requires_grad()
        # gt = torch.from_numpy(gt)
        # gt.requires_grad()
        # aaa = pred.long()* gt.long()
        # # print(aaa)
        # i = torch.sum(aaa)
        #
        # u = torch.sum(pred.long()) + torch.sum(gt.long())
        # print(i,u)
        # loss = (2 * i + eps) / (u + eps)
        # print('loss',loss)
        # loss.requires_grad()

        intersection = (pred * gt).sum()      # sum是计算对应重叠部分的和，并且是按照概率的
        # print('wobuxin:',pred.sum(),gt.sum())
        unionset = pred.sum() + gt.sum()   #计算并集，并且是按照预测的概率的计算，sum里头没有参数表示计算全部的和，
                                            # 也可以给参数，dim=0/1，行或列，但是我这里是计算整个batch的，就不用分行或列单独计算了
        # print('=====================')
        # print(intersection.requires_grad)
        # print(unionset.requires_grad)
        # print(pred.requires_grad)
        # print(gt.requires_grad)
        # print('intersection:',intersection)
        # print('unionset:',unionset)

        dice = (2 * intersection + eps) / (unionset + eps)   # 加上一个很小的数，以防除以0，没有object，并且也没有预测obj，这时候的dice已经是很高的，说明预测正确了，dice=1
        # print('loss',loss.requires_grad)
        # print('check interseciton and union{} / {} = {}'.format(i, u,loss))
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

    def diceCoeff(self,pred, gt):   # 这些都写成cpu的模式了，是没有梯度的
        eps = 1e-5

        # N = pred.shape[0] # N 是batch size
        i = np.sum(pred * gt)
        u = np.sum(pred) + np.sum(gt)
        loss = (2 * i + eps) / (u + eps)

        # print('check interseciton and union{} / {} = {}'.format(i, u,loss))
        return loss

        # pred = pred.view(N, -1)  # 后面512*512拉成一行

        # intersection = (pred * gt).sum(1)
        # unionset = pred.sum(1) + gt.sum(1)

        # print('intersection:',intersection)
        # print('unionset:',unionset)

        # loss = (2 * intersection + eps) / (unionset + eps)
        # print('=================',loss.sum() / N)
        # return loss.sum() / N  # batch size 求平均

    def SoftDiceLoss(self,output,gt):     #这两个是网络上的dice loss，抄下来想用来着，结果tensor不对，用不上，没有forward backward，没有梯度
        pred = output.data.cpu().numpy()
        target = gt.cpu().numpy()
        pred = np.argmax(pred, axis=1)  # 变成01234

        self.num_class = 5
        # 多个类别分别计算dice
        class_dice = []

        for i in range(1,self.num_class):  # 第1个类别到第5个类别，就是1,2,3,4桐花树，无瓣海桑，茳芏，秋茄
            temp_pred = copy.deepcopy(pred)
            temp_pred[temp_pred != i] = 0
            temp_target = copy.deepcopy(target)
            temp_target[temp_target != i] = 0

            # class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
            class_dice.append(self.diceCoeff(pred, target))

        mean_dice = sum(class_dice) / len(class_dice)  # 4个类别的dice求平均
        return 1 - mean_dice

    def Multi_loss(self,logit, target):
        n, c, h, w = logit.size()
        # print('logit  size:',logit.size())   # 8,5,512,512
        # print('target size:',target.size())   # 8,512,512
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        print('>>>>',logit.requires_grad,target.requires_grad)  # true false

        loss_ce = criterion(logit, target.long())

        if self.batch_average:
            loss_ce /= n
        print('wce loss:',loss_ce)

        # pred = logit.data.cpu().numpy()
        # target = target.cpu().numpy()
        # pred = np.argmax(pred, axis=1)  # 变成01234
        # print('pred1  :',np.unique(pred))

        pred = torch.argmax(logit,dim= 1)   # 对应预测的类别(0,1,2,3,4)
        print(pred.size)
        # print('======================================')
        # pred_p = torch.max(logit,dim = 1)[0]     # 这里的logit是有梯度的，但是梯度一定要是小数才有梯度，看下面的注释，之前想错了，
                                                # 之前直接用预测的类别01234来和对应的one hot数组算，肯定是错的。
                                                # 这里的pred_p是对应的每一列（5个概率）选最大的那个概率
                                                # 上面的pred 是记录最大概率的index，把各个类别的index找到写成one hot数组，和pred_p相乘，就能得到对应类别的预测概率，其他类别的就变成0了
        # 这里有一点错误，没有做softmax，d一列之内加和不为1，会有很大的值，算出来的dice loss会变成负数，是不对的

        log_predict = F.softmax(logit, dim=1)  # sogtmax 0-1
        # print('cccccc:',log_predict.size())
        # print('check 2:', torch.max(log_predict), torch.min(log_predict))
        pred_p = torch.max(log_predict, dim=1)[0]  # 每一个pixel的最大概率对应的概率，跟上面对应的类别对应

        # print('pred p:',pred_p)
        # print('00000000000',pred_p.size())
        # ppppp = pred_p[1]
        # print('1111111111111111',ppppp.size)
        # 还要拿到对应预测类别对应的概率
        # pred.requires_grad = True       # 报错：离谱……   pred.requires_grad = True RuntimeError: only Tensors of floating point dtype can require gradients
        # print(pred.requires_grad)  #false
        # print('pred2   :',pred)
        # print('pred size:',pred.shape)  # 8,512,512
        # print('===========================')
        # print(np.unique(pred))    # [0 1 2 3 4]
        # print('===========================')
        self.num_class = 5
        # 多个类别分别计算dice
        class_dice = []
        criterion2 = My_Loss()
        if self.cuda:
            criterion2 = criterion2.cuda()
        # print('target size:',target.size())
        for i in range(1, self.num_class):  # 第1个类别到第5个类别，就是1,2,3,4桐花树，无瓣海桑，茳芏，秋茄
            temp_target = torch.zeros(target.size())
            temp_target[target == i] = 1
            # print(i,'targetsum:',temp_target.sum())    # 还是会存在如果target全为0的情况
            # if temp_target.sum() == 0:      # dice loss中加了一个eps，说明是有考虑到当target全为0的情况的，所以不需要另外考虑这个情况。
            #     continue                    # 否则如果这个batch刚好全都是空白的没有object的图像，dice loss就为0了，而且是int的0，因为没有给到tensor中，这就会没有梯度了，反向传播时会报错

            temp_pred = torch.zeros(target.size()).cuda()   # 这里的cuda是必须的
            temp_pred[pred == i] = 1

            temp_pred *= pred_p   #该pixel预测类别对应的概率
            # temp_pred.requires_grad = True


            tttt = criterion2(temp_pred.cuda(), temp_target.cuda())  # 这里的cuda也是必须把，把float tensor转成cuda的float tensor
            # print('tttt:',tttt)
            class_dice.append(tttt)
        print('class dice:',class_dice)
        # if len(class_dice) == 0:   # 这是一张空白的图片
        #     return 0
        mean_dice = sum(class_dice) / len(class_dice)  # 4个类别的dice求平均
        loss_dice = 1 - mean_dice   # 注意是1减去整个batch的mean_dice之后再求batch的平均
        loss_dice /= n  # 除以batch size大小，求平均每张图的dice loss，前面的wce loss也是求平均每张图的wce loss

        print('dice loss:',loss_dice)
        loss= loss_ce + loss_dice       # 现在两个loss的比例是1比1，并且两个loss都是tensor，有梯度的
        # return loss_ce,loss_dice
        # return loss_dice
        print('return loss:',loss)
        return loss



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




