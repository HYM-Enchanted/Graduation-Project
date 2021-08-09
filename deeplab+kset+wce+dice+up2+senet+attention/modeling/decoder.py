import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.se_module import SELayer

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv_low = nn.Sequential(nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
                                      BatchNorm(48),
                                      nn.ReLU())


        # todo 上采样，使用反卷积恢复feature尺寸
        self.up_conv = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1),
                                     # BatchNorm(128),
                                     # nn.ReLU(),
                                     # nn.Dropout(0.5),
                                     nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
                                     BatchNorm(64),
                                     nn.ReLU(),
                                     # nn.Dropout(0.5))
                                     )
        # self.up_conv = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                              # BatchNorm(128),
        #                              # nn.ReLU(),
        #                              # nn.Dropout(0.5),
        #                              nn.ConvTranspose2d(256, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                              BatchNorm(192),
        #                              nn.ReLU(),
        #                              # nn.Dropout(0.5))
        #                              )
        # todo 对双线性插值的结果改变channel数量
        self.conv_inter = nn.Sequential(nn.Conv2d(256,192,kernel_size=1,stride=1,),
                                        BatchNorm(192),
                                       nn.ReLU())

        # self.conv_inter = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, ),
        #                                 BatchNorm(64),
        #                                 nn.ReLU())

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       # nn.Conv2d(256, 32, kernel_size=1, stride=1))  #    # 这里维度变成32，然后做双线性插值上采样变成512*512后，再根据mask的信息？mask的信息是不是该加的早一点
                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))   # 这里output的tensor就已经是5个类了……阿这……
                                        # nn.Conv2d(256, 4, kernel_size=1, stride=1))   # 这里output的tensor就已经是5个类了……阿这……
        self._init_weight()


        # todo selayer
        self.se = SELayer(304,16)  # 304正好可以被16整除=19，也就是这个通道注意力模块是先把304channel降维成19然后再回复成304的

        self.se_aspp = SELayer(256,16)

        # self.se_interpolate = SELayer(256,16)

    def forward(self, x, low_level_feat):  # x是aspp最后的输出结果
        # print('decoder start')
        # print('decoder input size x:',x.size())    # torch.Size([1, 256, 32, 32])  x是aspp的output
        # print('decoder input size llf:',low_level_feat.size())  # torch.Size([1, 256, 128, 128])
        low_level_feat = self.conv_low(low_level_feat)
        # print('check 3:',low_level_feat.size())  # torch.Size([1, 48, 128, 128])
        # x为aspp的output
        # todo 把aspp的output做se
        x = self.se_aspp(x)

        # todo 两次反卷积，上采样4倍
        x1 = self.up_conv(x)   # 32 - 128
        # 直接使用双线性插值恢复图像尺寸
        x2 = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)

        # todo 在双线性插值上采样结束之后的特征进行同特注意力机制操作
        # x2 = self.se_interpolate(x2)   #256,128,128   # 这里加了也没有秋茄了...

        x2 = self.conv_inter(x2) # 256 -> 192
        # print('check 4:',x2.size())               #现在是：torch.Size([1, 128, 128, 128])， 以前是：torch.Size([1, 256, 128, 128])
        # x = torch.cat((x, low_level_feat), dim=1)
        x = torch.cat((x1,x2, low_level_feat), dim=1)
        del x1,x2,low_level_feat
        # print('check 5:',x.size())               # torch.Size([1, 304, 128, 128])   # low level feat只占48份，占比比较少
        #todo 在这里加入se模块，通道注意力模块，因为接下来的last conv模块会把tensor变成5个channel，就已经是分好类了
        x = self.se(x)

        x = self.last_conv(x)
        # print('check 6:',x.size())               # torch.Size([1, 5, 128, 128])
        # print('decoder end')
        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)