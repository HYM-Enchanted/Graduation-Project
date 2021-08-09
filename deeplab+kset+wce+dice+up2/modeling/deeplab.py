import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=5,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

        # todo 上采样采用反卷积恢复decoder 的 output feature尺寸
        self.up_conv = nn.Sequential(nn.ConvTranspose2d(5,5,kernel_size=3,stride=2,padding=1,output_padding=1),
                                     nn.ConvTranspose2d(5,5,kernel_size=3,stride=2,padding=1,output_padding=1),
                                     BatchNorm(5),
                                     nn.ReLU(),
                                     nn.Dropout(0.5))

        # todo 对双线性插值的结果改变channel的数量
        self.conv_inter = nn.Sequential(
                                        BatchNorm(5),
                                        nn.ReLU())

        self.output_conv = nn.Sequential(nn.Conv2d(10, 8, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(8),
                                       nn.ReLU(),
                                       nn.Conv2d(8, num_classes, kernel_size=1, stride=1))

    def forward(self, input):
        # print('deeplab')
        x, low_level_feat = self.backbone(input)
        # print('check x',x.size())                  # torch.Size([1, 2048, 32, 32])
        # print('check llf:',low_level_feat.size())  # torch.Size([1, 256, 128, 128])
        x = self.aspp(x)
        # print('aspp',x.size())                     # torch.Size([1, 256, 32, 32])
        x = self.decoder(x, low_level_feat)   # 128
        # print('x decoder',x.size())                # torch.Size([1, 5, 128, 128])
        # x1 = self.up_conv(x)     # 对decoder的结果上采样两次 32
        # print('check 2:', x1.size())               # torch.Size([1, 5, 512, 512])
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  #这里还一下子上采样4倍了
        # print('check 3:', x2.size())               #torch.Size([1, 128, 512, 512])
        # x2 = self.conv_inter(x2)   # 32
        # print('check 4:', x2.size())               #torch.Size([1, 5, 512, 512])
        # x = torch.cat((x1,x2),dim=1)
        # print('check 5:', x.size())                #torch.Size([1, 10, 512, 512])
        # del x1,x2,low_level_feat
        # x = self.output_conv(x)
        # print('output size :',x.size())            # torch.Size([1, 5, 512, 512])
        return x



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    # model = DeepLab(backbone='mobilenet', output_stride=16)
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    # input = torch.rand(1, 3, 513, 513)
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())  # torch.Size([1, 5, 512, 512])  5是num classes


