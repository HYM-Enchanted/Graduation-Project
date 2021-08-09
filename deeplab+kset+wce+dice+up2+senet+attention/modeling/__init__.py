import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf,suppress=True)#使print大量数据不用符号...代替而显示所有
# 尝试一下卷积的输入全为0或1 ，卷积过后会输出什么东西

class Conv01():
    def __init__(self):
        input = Image.open('F:\Google download/0105output\ppp-2000/170_mask.png')
        input = np.array(input)  # 不行哦还要变成0和1
        input = self.encode_segmap2(input)
        # plt.imshow(input)
        # plt.grid()
        # plt.show()
        # exit()

        input = np.array([[input]])
        print(input.shape)
        input = np.stack((input,input,input,input))
        print(input.shape)
        input = torch.from_numpy(input).float()  # 变成tensor
        print('input shape',input.shape)



        exit()

        # todo 如果直接用下采样4倍？
        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=4,padding=1)
        x_down = self.maxpool4(input)
        print(x_down.shape)
        print(x_down[0,0,100//4:116//4,384//4:400//4])
        print()


        BatchNorm = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn2 = BatchNorm(64)

        x = self.forward(input)

    def forward(self,input):
        print(input[0,0,100:116,384:400])
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        print(x.shape)
        print(x[0,0,100//4:116//4,384//4:400//4])
        # a = x[0,0,100//2:110//2,390//2:400//2]
        # print(a)
        # a = a.detach().numpy()
        # print(a)

    def get_mangrove_labels2(self):
        return np.asarray([
            [0, 0, 0],  # 背景
            [244, 244, 0],  # 红树林
            # [255,0,0],     #无瓣海桑
            # [0,255,0],     #茳芏
            # [0,255,255]    #秋茄
        ])

    def encode_segmap2(self, mask):  # 从颜色到01234
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        # for ii, label in enumerate(get_pascal_labels()):
        for ii, label in enumerate(self.get_mangrove_labels2()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask


# Conv01()

