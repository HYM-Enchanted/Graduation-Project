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

        log_softmax_func = nn.LogSoftmax(dim=1)   #这里output就是4个类别
        nll_loss_func = nn.NLLLoss(ignore_index=-1)
        logit_zeros = torch.zeros(logit.shape)
        logit_zeros = logit_zeros.cuda()
        logit_sm = log_softmax_func(logit[:,1:,:,:])
        print('logit log softmax:', logit_sm[0, :, 10, 10])
        # print('logit sm shape:',logit_sm.shape)   # 但愿他是batch size * 4 * 512 * 512
        # print('sdfsdfdsf',logit_sm[0,0,:4,:4])   # 这个只是softmax

        logit_zeros[:,1:,:,:] = logit_zeros[:,1:,:,:] + logit_sm
        logit_sm = logit_zeros
        pred = torch.argmax(logit_sm[:,1:,:,:], dim=1)  # 对应的类别，肯定不会是背景类
        print('pred shape:',pred.shape)   # batch size * 512 * 512
        print('check leibie ceeeee:', pred[0, :5, :5])   # 这里的类别比下面的类别小于1，因为这里只有4个类
        # todo 去掉背景类，把背景类的地方的概率值都改成1，计算loss的时候这个像素点的loss就是0了
        # todo 首先要把mask变成4维的
        # for i in range(mask.shape[0]):
        #     aa = torch.stack([mask[i],mask[i],mask[i],mask[i]],dim=0)
        #     aa = torch.unsqueeze(aa,dim= 0 )
        #     # print('check aa', aa.shape)
        #     if i ==0:
        #         batch_mask = aa
        #     else:
        #         batch_mask = torch.cat((batch_mask,aa),dim=0)
        # # print('check bb:', batch_mask.shape)
        # print('batch shape',batch_mask.shape)

        # logit_sm[batch_mask == 0] = 1    #这里有inplace操作，
        if self.cuda:
            nll_loss_func = nll_loss_func.cuda()

        # 我这个4分类，那么target还要-1，然后如果是背景类的区域，概率全给0？
        # 不能减1 ，减1会存在-1的地方……
        # print('check target class 01234：',torch.unique(target))
        # 那就是把背景类的地方概率都给1，那么指定都是
        # 这个target是index的嘛？那么因为是4分类，所以还是要-1，然后是-1的地方改成0就好了，反正也是背景类的像素点计算出来的loss也是为0
        target_ce = copy.deepcopy(target)
        target_ce -= 1    # 把背景类做-1，其他红树林为0123,这是计算ce loss的标签，和底下计算dice loss的标签是不同的
        # target_ce[target_ce < 0] = 0
        # print('check target 0123：', torch.unique(target))
        # todo target不要动，直接不计算标签为0的地方的loss
        loss_ce = nll_loss_func(logit_sm,target_ce.long())   # 这里ignore -1背景类

        if self.batch_average:
            loss_ce /= n

        print('ce loss:',loss_ce)

        # todo 为了确保我这么计算是真实有效的，要检查一下带上背景类的是否计算得到的loss会不一样
        test_logit = log_softmax_func(logit)
        loss_ce0 = nll_loss_func(test_logit, target.long())
        if self.batch_average:
            loss_ce0 /= n

        print('ce000 loss:', loss_ce0)

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
        # print('logit sm shape:', logit_sm.shape)  # 但愿他是batch size * 4 * 512 * 512
        print('logit gailu:',logit_sm[0,:,10,10])
        logit_zeros = torch.zeros(logit.shape)
        logit_zeros = logit_zeros.cuda()
        logit_zeros[:, 1:, :, :] = logit_zeros[:, 1:, :, :] + logit_sm
        logit_sm = logit_zeros
        print('check last:',logit_sm[0,:,10,10])
        pred = torch.argmax(logit_sm, dim=1)  # 对应的类别，肯定不会是背景类
        # print('pred shape:',pred.shape)   # batch size * 512 * 512
        print('check leibie:',pred[0,:5,:5])
        # print('check mask 111111:',torch.sum(pred == 1))
        # print('check mask 000000:',torch.sum(pred == 0))



        #todo 去掉背景类的像素点的位置
        # print('pred index:',pred.shape)
        mask = mask.cuda()
        pred *= mask.long()

        # print('check mask out bg 1111:',torch.sum(pred == 1))
        # print('check mask out bg 0000:', torch.sum(pred == 0))
        pred_p = torch.max(logit_sm,dim=1)[0]   # 对应类别的概率,

        # print('check mask===============',torch.sum(mask))

        self.num_class = 5    # 4个红树林的类别
        # 多个类别分别计算dice
        class_dice = []

        criterion2 = My_Loss()
        if self.cuda:
            criterion2 = criterion2.cuda()

        # rate = []

        # todo 第一种方案：按照ce loss的weight，计算那个weight到0-1之间
        # dice_weight = self.weight[:]
        # print('check dice weight', dice_weight)
        # balance_weight = dice_weight / dice_weight.sum()
        # print('softmax dice weight', balance_weight)

        # todo 第二种方案是根据每个batch里面各个class的数量做balanced，还没做
        # target中对应的红树林还是1234，背景类才是0
        # todo 不知道为毛不行，那就全都减1
        for i in range(1,self.num_class):  # 一共就4个类别了现在，就是1,2,3,4桐花树，无瓣海桑，茳芏，秋茄
            temp_target = torch.zeros(target.size())   # 512* 512
            temp_target[target == i] = 1   #对应类别的像素点的位置置为1
            # rate.append(temp_target.sum())
            temp_pred = torch.zeros(target.size()).cuda()

            temp_pred[pred == (i-1)] = 1   # 选到对应的类别，造一个one，对应的位置

            temp_pred *= pred_p   #该pixel预测类别对应的概率

            class_dice.append(criterion2(temp_pred.cuda(), temp_target.cuda()))

        # print('rate:',rate,'\n')
        # print('\n====================')
        # avg_class_dice = [class_dice[i] * 0.25 for i in range(len(balance_weight))]
        # print('class dice:',avg_class_dice,'\n')
        # print('sum class dice:',sum(avg_class_dice),'\n')
        ###mean_dice = sum(class_dice) / len(class_dice)  # 4个类别的dice求平均  ###三个#的是以前的dice loss

        # rate = [r/sum(rate) for r in rate]
        # print('precent:',rate,'\n')
        ####balance_class_dice = [class_dice[i] * balance_weight[i].item() for i in range(len(balance_weight))]

        # print('balance dice',balance_class_dice,'\n')
        # print('sum balance dice:', sum(balance_class_dice),'\n')
        # print('============================')
        ####balanced_dice = sum(balance_class_dice)

        ####loss_balanced_dice = (1 - balanced_dice)/n   # 对整个batch求平均
        # print('balance dice loss:',loss_balanced_dice)
        ###loss_dice = 1 - mean_dice   # 注意是1减去整个batch的mean_dice之后再求batch的平均
        ###loss_dice /= n    # 除以batch size大小，求平均每张图的dice loss，前面的wce loss也是求平均每张图的wce loss
        # print('avg dice loss:',loss_dice)
        # print('wce loss:',loss_ce,'\n')
        # print('return loss:',loss_ce,'\n',loss_dice,'\n')   #还是想打印一下loss
        # loss= loss_ce + loss_dice
        # return loss_ce,loss_dice   # 分别返回loss，可以画tensorboard
        ####return loss_ce,loss_balanced_dice   # 分别返回loss，可以画tensorboard
        # return loss_dice   # 检查只有一个dice loss有没有梯度，能不能反向传播，能！！！
        # print('return loss:',loss_ce,'\n',loss_dice)
        # return loss
        return loss_ce,class_dice



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




