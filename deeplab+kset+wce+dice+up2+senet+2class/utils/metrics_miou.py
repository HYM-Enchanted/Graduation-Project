import numpy as np
np.set_printoptions(threshold=np.inf)

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):   # class acc
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print('class acc for 5 class:',Acc)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        class_MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        print('each class miou:',class_MIoU)     # 每个类别的miou
        MIoU = np.nanmean(class_MIoU)   #取平均（跳过0值求平均）
        return MIoU,class_MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        print('frequency_weighted:',freq)   # 按权重的iou，freq是权重
        # print('miou:',iu)  # miou是一样的
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)    # 选择几个类别，这里这句话就是选择所有的类别
        # todo 如果我只需要选择其中一个类别，那只需要让mask在gt等于该类别的时候为true
        # mask = gt_image == 0   # 我只选择了一个类别
        # print('mask',mask.shape)  # batch size ,512,512
        # print('mask:',np.unique(pre_image))
        # print('count :',np.sum(pre_image == 1),np.sum(pre_image == 2),np.sum(pre_image == 3),np.sum(pre_image == 4))
        # print('==================================================================')
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # print('label',np.unique(label))
        count = np.bincount(label, minlength=self.num_class**2)   #np.bincount 统计每个索引出现的次数，返回的是一个array
        # print('count:',count)   # 是一个5*5=25的一维array
        confusion_matrix = count.reshape(self.num_class, self.num_class)    # reshape成一个5*5的矩阵
        # print('===================================================================')
        # print('confusion_matrix',confusion_matrix)   #这是每个batch的

        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)


    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




