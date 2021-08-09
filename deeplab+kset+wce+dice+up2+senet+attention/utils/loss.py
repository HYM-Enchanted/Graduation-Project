import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
# from .dice_loss_metrics import *

# todo 11月14号
# todo 准备根据上一个epoch的miou来给定weight
# 但是需要上一个epoch的miou，所以不好传入，干脆直接返回class dice到train中再计算。


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


    def Multi_loss(self,logit, target,mask):
        n, c, h, w = logit.size()
        # print('ce loss weight',self.weight)   #ce loss weight tensor([ 1.4607, 20.7817, 40.7756, 44.3521, 50.2720])

        # criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
        #                                 size_average=self.size_average)
        #
        # if self.cuda:
        #     criterion = criterion.cuda()
        #
        # loss_ce = criterion(logit, target.long())

        # print('self weight:',self.weight[1:])  #self weight: tensor([ 2.1202,  5.1093,  5.3738, 18.1546])
        """因为这里把背景类弄成-1了，-1的类别不占weight的"""
        log_softmax_func = nn.LogSoftmax(dim=1)   #这里output就是五维的 batch size * 5 * 512 * 512
        nll_loss_func = nn.NLLLoss(ignore_index=-1,weight=self.weight[1:])    #加权重！！
        # logit_zeros = torch.zeros(logit.shape)
        # logit_zeros = logit_zeros.cuda()
        logit_sm = log_softmax_func(logit[:,1:,:,:])   # 计算softmax的时候（或者log softmax）的时候不要背景类，免得他把概率全拉过去了，直接4分类
        # 这里也只是0123啊？？！！哦不是这个是log softmax不是类别
        # logit_zeros[:,1:,:,:] = logit_zeros[:,1:,:,:] + logit_sm    # 在这里补上第0个背景类，这个类别的概率全为0，因为后面计算loss的时候是不计算这个的
        # logit_sm = logit_zeros

        if self.cuda:
            nll_loss_func = nll_loss_func.cuda()

        # todo 把target也就是标签均减去1，然后把-1也就是背景类在计算loss的时候ignore掉，这个点不参与计算，只做4分类
        target_ce = copy.deepcopy(target)
        target_ce -= 1    # 把背景类做-1，其他红树林为0123,这是计算ce loss的标签，和底下计算dice loss的标签是不同的
        # 既然-1不占weight，那么我的weight的维度为1*4即可
        # todo target不要动，直接不计算标签为0的地方的loss
        loss_ce = nll_loss_func(logit_sm,target_ce.long())   # 这里ignore -1背景类,,,这里之前是没有用上减去1的类别啊我的老天。。。

        if self.batch_average:
            loss_ce /= n

        # print('ce loss:',loss_ce)

        #todo 为了确保我这么计算是真实有效的，要检查一下带上背景类的是否计算得到的loss会不一样，检查成功
        # test_logit = log_softmax_func(logit)
        # loss_ce0 = nll_loss_func(test_logit,target.long())
        # if self.batch_average:
        #     loss_ce0 /= n
        #
        # print('ce000 loss:',loss_ce0)


        # 以上得到了wce loss，以下计算dice loss，然后结合，注意比例
        # todo 计算 dice loss 了
        # pred = torch.argmax(logit,dim= 1)   # 对应预测的类别，这里只有4个类别,又改回5个类别了，先把ce loss搞出来
        # # print('check 1:',torch.max(logit),torch.min(logit))
        # log_predict = F.softmax(logit,dim=1)   # sogtmax 0-1 , 对应的概率？
        # print('cccccc:',log_predict.size())
        #
        # # print('check 2:', torch.max(log_predict), torch.min(log_predict))
        # pred_p = torch.max(log_predict ,dim = 1)[0]  # 每一个pixel的最大概率对应的概率，跟上面对应的类别对应
        # print("yuayudyafa:",torch.max(pred_p),torch.min(pred_p))

        # todo 要计算出来各个类别的概率，这里之前的不是概率，而是log（概率）的值
        softmax = nn.Softmax()
        logit_sm = softmax(logit[:,1:,:,:])   # softmax 之后是概率
        logit_zeros = torch.zeros(logit.shape)
        logit_zeros = logit_zeros.cuda()
        logit_zeros[:, 1:, :, :] = logit_zeros[:, 1:, :, :] + logit_sm
        logit_sm = logit_zeros
        # print('check dice loss classes:',logit_sm.shape)   #check dice loss classes: torch.Size([8, 5, 512, 512])
        pred = torch.argmax(logit_sm, dim=1)  # 对应的类别，肯定不会是背景类
        # print('check classes:',torch.unique(pred))   #check classes: tensor([1, 2, 3, 4], device='cuda:0')
        #todo 去掉背景类的像素点的位置
        mask = mask.cuda()
        pred *= mask.long()

        pred_p = torch.max(logit_sm,dim=1)[0]   # 对应类别的概率,肯定不是背景类的概率

        self.num_class = 5    # 4个红树林的类别
        # 多个类别分别计算dice
        class_dice = []

        criterion2 = My_Loss()
        if self.cuda:
            criterion2 = criterion2.cuda()

        # todo 第二种方案是根据每个batch里面各个class的数量做balanced，还没做
        # target中对应的红树林还是1234，背景类才是0
        # todo 不知道为毛不行，那就全都减1
        for i in range(1,self.num_class):  # 一共就4个类别了现在，就是1,2,3,4桐花树，无瓣海桑，茳芏，秋茄
            temp_target = torch.zeros(target.size())   # 512* 512
            temp_target[target == i] = 1   #对应类别的像素点的位置置为1
            # rate.append(temp_target.sum())
            temp_pred = torch.zeros(target.size()).cuda()

            temp_pred[pred == i] = 1   # 选到对应的类别，造一个one，对应的位置

            temp_pred *= pred_p   #该pixel预测类别对应的概率

            class_dice.append(criterion2(temp_pred.cuda(), temp_target.cuda()))

        return loss_ce,class_dice



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




