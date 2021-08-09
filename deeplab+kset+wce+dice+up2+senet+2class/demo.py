#
# demo.py
#
import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image
from utils.metrics_miou import Evaluator
import cv2
# import torchvision.transforms as transforms
# 计算预测的miou，输入需要有图片和label（target）


# https://github.com/pytorch/pytorch/issues/229
import torch
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, default = '/mnt/mangrove_test/whole_test_npz', help='image to test')  # 预测图片的path
    parser.add_argument('--label-path', type=str, default = '/mnt/mangrove_test/whole_test_npz', help='image to test')  #预测图片的label的path
    parser.add_argument('--out-path', type=str,default = '/mnt/mangrove_test/test_output_1105', help='mask image to save')  # 输入的path
    parser.add_argument('--flip',type=str,default= None,help='flip the images')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='/mnt/project/deeplab_mobilenet_npz+kset/run/mangrove/deeplab-resnet/model_best.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='mangrove',
                        choices=['pascal', 'coco', 'cityscapes','mangrove'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()
    model = DeepLab(num_classes=args.num_classes,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time
    print("model load time is {}".format(model_load_time))
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # todo :增加计算精度的部分
    evaluator = Evaluator(args.num_classes)

    composed_transforms = transforms.Compose([
        #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    start_time = time.time()

    for name in os.listdir(args.in_path):
        s_time = time.time()
        #image = Image.open(args.in_path+"/"+name).convert('RGB')
        image = np.load(os.path.join(args.in_path,name))
        image = image['image'][:,:,1:]

        # image = np.flip(image,0)
        image = np.flip(image,1)   # 0,1,0+1 三种

        label_url = os.path.join(args.label_path,name)
        label_url = label_url.replace('npz','png')
        target = Image.open(label_url)

        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)
        # print('check output tensor:',output.size())  # output的维度torch.Size([1, 5, 512, 512])
        # print('check grid tensor:',output[:3].size()) #torch.Size([1, 5, 512, 512])
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),  # 如果是4通道的话，output[:4]??
                                3, normalize=False, range=(0, 255))

        # 计算评价指标
        target = np.array(target)
        # target = np.flip(target, 0)  # 标签翻转
        target = np.flip(target, 1)  # 标签翻转
        target = np.array([target])
        # print(target.shape) #(1, 512, 512)

        pred = output.data.cpu().numpy()
        pred = np.argmax(pred,axis=1)
        # print(pred.shape) # (1, 512, 512)
        evaluator.add_batch(target,pred)   # 所有的图片一起计算acc，miou等

        # print("type(grid) is: ", type(grid_image))  #type(grid) is:  <class 'torch.Tensor'>
        # print("grid_image.shape is: ", grid_image.shape)  #torch.Size([3, 512, 512])

        # grid_image = flip(grid_image,1)   # 对应np.flip 0
        grid_image = flip(grid_image,2)   #对应np.flip 1

        save_image(grid_image,args.out_path+"/"+"{}_mask.png".format(name[0:-4]))

        u_time = time.time()
        img_time = u_time-s_time
        print("image:{} time: {} ".format(name,img_time))


    end_time = time.time()
    print('predict all images {} ,use time : {} s\n'.format(len(os.listdir(args.in_path)),end_time - start_time))
    # todo：预测了所有的图片并保存好了预测结果，计算评价指标
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU, class_miou = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    # print('5 class miou:',class_miou)  # 在调用evaluator里头就会打印了


    print("image save in in_path.")
if __name__ == "__main__":
   main()

# python demo.py --in-path your_file --out-path your_dst_file
