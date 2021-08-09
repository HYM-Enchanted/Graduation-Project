# 把output就开始保存，不做softmax也不做别的,也不要画mask了，因为是4分类，后处理拿下来之后我在做，输入mask的信息。

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
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image

#1116: 这是备份的预测的代码，是现在的训练输入样式（npz，4通道，自己根据网络是几通道保留几通道的数据）
# 在demo.py那边改成可以计算miou的（tp，fp）

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, default='/mnt/mangrove_cut0/only-mangrove', help='image to test')
    parser.add_argument('--out-path', type=str, default='/mnt/mangrove_cut0/only-output',
                        help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str,
                        default='/mnt/code/deeplab+kset+wce+dice+up2+senet+attention/run/mangrove/cut0-4classes/model_best.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='mangrove',
                        choices=['pascal', 'coco', 'cityscapes', 'mangrove'],
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
    model_load_time = model_u_time - model_s_time
    print("model load time is {}".format(model_load_time))

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # if not os.path.exists(os.path.join(args.out_path,'/predict_npy/')):
    #     os.makedirs(os.path.join(args.out_path,'/predict_npy/'))

    # if not os.path.exists(os.path.join(args.out_path,'/predict_npy/')):
    #     os.makedirs(os.path.join(args.out_path,'/predict_npy/'))

    composed_transforms = transforms.Compose([
        # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    start = time.time()
    for name in os.listdir(args.in_path):
        s_time = time.time()
        # image = Image.open(args.in_path+"/"+name).convert('RGB')
        image = np.load(os.path.join(args.in_path, name))
        image = image['image'][:, :, 1:]
        # image = image['image'][:, :, :3]
        target = image
        sample = {'image': image, 'label': target,'mask':target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)
        # print('check output tensor:',output.size())  # output的维度torch.Size([1, 5, 512, 512])
        # print('check grid tensor:',output[:3].size()) #torch.Size([1, 5, 512, 512])
        # todo 这里就不画图了
        # grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
        #                        3, normalize=False, range=(0, 255))
        #
        # save_image(grid_image, args.out_path  + "/{}_mask.png".format(name[0:-4]))

        # output = F.softmax(output)  #变成概率，数组的大小会不会小一点
        #
        # predict_p = torch.max(output,dim =1)[0]  # 获得对应的类别
        # predict_index = torch.argmax(output,dim=1)
        # save_output =
        # print('predict:',predict_index.size(),'\n')
        # print(predict_p.size(),'\n')
        # save_output = np.array(predict_p)+np.array(predict_index)
        save_name = os.path.join(args.out_path,'{}_output.npy'.format(name[0:-4]))
        np.save(save_name,output)

        u_time = time.time()
        img_time = u_time - s_time
        print("image:{} time: {} ".format(name, img_time))
        # save_image(grid_image, args.out_path)
        # print("type(grid) is: ", type(grid_image))
        # print("grid_image.shape is: ", grid_image.shape)
    print("image save in in_path.")
    end = time.time()
    print('predice {} images use time: {}'.format(len(os.listdir(args.in_path)),end - start))

if __name__ == "__main__":
    main()

# python demo.py --in-path your_file --out-path your_dst_file
