import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

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

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  #自适应平均池化函数，直接pool成了1×1
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)   # 这里的dropout是0.5
        self._init_weight()

    def forward(self, x):
        # print('aspp input x size:',x.size())  #torch.Size([1, 2048, 32, 32])
        x1 = self.aspp1(x)
        # print('check 2:',x1.size())            #torch.Size([1, 256, 32, 32])
        x2 = self.aspp2(x)
        # print('check 2:', x2.size())           #torch.Size([1, 256, 32, 32])
        x3 = self.aspp3(x)
        # print('check 3:', x3.size())           #torch.Size([1, 256, 32, 32])
        x4 = self.aspp4(x)
        # print('check 4:', x4.size())           #torch.Size([1, 256, 32, 32])
        x5 = self.global_avg_pool(x)
        # print('check 5:', x5.size())           #torch.Size([1, 256, 1, 1])    # 32x32直接pool成1×1？？？
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # bilinear 双线性插值？插值前后的size 维度没变，但是数值是变了的
        # print('check 6:', x5.size())           #torch.Size([1, 256, 32, 32])   # 然后把1×1通过双线性插值成32×32？
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)   # torch tensor 的concat起来
        # print('check 7',x.size())             #torch.Size([1, 1280, 32, 32])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('check 8',x.size())             #torch.Size([1, 256, 32, 32])
        # print('aspp end')
        return self.dropout(x)  # 上面选了dropout的系数为0.5

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)