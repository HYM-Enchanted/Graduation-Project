import argparse
import os
import random

import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
# from utils.metrics import Evaluator
from utils.metrics_miou import Evaluator    # 让他调用我改了的这个metrics_miou
# from utils.new_loss import SoftDiceLoss

from torch.utils.data import DataLoader
# wce loss + dice loss，dice loss的比例按照前一个epoch的miou来计算
# 修改dice loss的比例，不然前五个epoch他就相当于是废了


class Trainer(object):
    def __init__(self, epoch, index_list, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.epoch = epoch
        self.index_list = index_list

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(self.epoch,
                                                                                             self.index_list, args,
                                                                                             **kwargs)
        # print('train loader:',len(self.train_loader))   #train loader: 2464

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        # self.best_pred = best_pred
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        self.batch_size = self.args.batch_size

        # self.balance_m_weights = torch.from_numpy(np.array([0.25,0.25,0.25,0.25]).astype(np.float32))
        self.balance_m_weights = np.array([0.25,0.25,0.25,0.25])

    def cal_weights(self,mious):
        # 根据前一次的miou计算weight，输入前一个epoch的4个类别的miou，然后计算比例
        # 第0个epoch没有前面的miou，直接把weight设置成0.25,0.25,0.25,0.25即可
        mious = np.array(mious)
        weights = []
        for m in mious:
            weights.append(1 / (np.log(1.02 + (m / sum(mious)))))
        balanced_mious_Weight = [weights[i] / sum(weights) for i in range(len(weights))]
        # balanced_mious_Weight = torch.from_numpy(balanced_mious_Weight.astype(np.float32))
        return balanced_mious_Weight



    def training(self, epoch):
        # print('train self .best pred',self.best_pred)
        train_loss = 0.0

        train_ce_loss,train_dice_loss = 0.0,0.0

        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        # x = 0.1 * (epoch // 5)   # 感觉加的比例有点太大，加上了这个dice loss之后，miou反而再降了
        # x = 0.05 * (epoch // 2)   # 还是一样的
        # 适当增大dice loss的比例
        # if epoch < 2:
        #     x = 0
        # elif epoch < 5:
        #     x = 0.2
        # elif epoch < 10:
        #     x = 0.5
        # elif epoch < 20:
        #     x = 0.8
        # else:
        #     x = 1   # 但是不要让dice loss的比例占比太大？

        # dice loss占的比例是previous best好了
        # x = np.float(self.best_pred)
        # print('dice loss weight',self.best_pred)
        # 可以尝试一下让dice loss 的占比大于wce loss。但是从哪一个epoch开始呢？
        # if epoch < 2:
        #     x = 0
        # elif epoch <5 :
        #     x = 0.5
        # elif epoch < 10:
        #     x = 1
        # elif epoch < 15:
        #     x = 1.5
        # elif epoch < 20:
        #     x = 2
        # else:
        #     x = 3

        # todo 要让wce loss和dice loss的加权之和为1
        if epoch < 2:
            x = 0
        elif epoch < 5:
            x = 0.25
        elif epoch < 10:
            x = 0.5
        elif epoch < 15:
            x = 0.6
        elif epoch < 20:
            x = 0.7
        else:
            x = 0.8

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            # print('check output size :',output.shape)  #torch.Size([8, 5, 512, 512])
            # loss_ce,loss_dice = self.criterion(output, target)  # 一个batch的loss，然后做反向传播
            # 这里变成return ce loss和class dice
            loss_ce,class_dice = self.criterion(output,target)
            # print('class dice:',class_dice,'\n')
            # avg_dice_loss = (1 - (sum(class_dice) / len(class_dice))) / self.batch_size
            # print('avg dice loss', avg_dice_loss,'\n')
            # todo 根据class dice 和balance miou weight计算dice loss
            balance_class_dice = [class_dice[i] * self.balance_m_weights[i] for i in range(len(class_dice))]

            # print('balance class dice:',balance_class_dice)
            loss_dice = (1 - sum(balance_class_dice)) / self.batch_size
            # print('ce loss',loss_ce,'\n')
            # print('dice loss',loss_dice)
            # todo 加和ce loss 和dice loss作为总的loss，后续可以给予不同的比例加和
            loss = (1-x) * loss_ce  + x * loss_dice  # 逐步加上dice loss的权重

            loss.backward()    # 每一个batch都做反向传播
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.4f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)  #这里打出的loss只包含ce loss
            train_ce_loss += loss_ce.item()
            self.writer.add_scalar('train/batch_ce_loss',loss_ce.item(),i + num_img_tr * epoch)
            train_dice_loss += loss_dice.item()
            self.writer.add_scalar('train/batch_dice_loss',loss_dice.item(),i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/total_ce_loss_epoch', train_ce_loss, epoch)   # 这些地方没有除以4，这是对应val的4倍，是没关系的
        self.writer.add_scalar('train/total_dice_loss_epoch', train_dice_loss, epoch) # 这里也没有除以4
        self.writer.add_scalar('train/loss_per_5set', train_loss / 4, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.5f, Loss per 5 set : %.5f, CE Loss : %5f, Dice Loss : %5f' % (train_loss, train_loss / 4,train_ce_loss,train_dice_loss))
        print('dice rate:',x)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):  # 验证集，每10个epoch求平均
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        test_ce_loss,test_dice_loss = 0.0,0.0
        # x = 0.1 * (epoch // 5)
        # x = 0.05 * (epoch // 2)
        # if epoch < 2:
        #     x = 0
        # elif epoch < 5:
        #     x = 0.2
        # elif epoch < 10:
        #     x = 0.5
        # elif epoch < 20:
        #     x = 0.8
        # else:
        #     x = 1   # 但是不要让dice loss的比例占比太大？

        # x = np.float(self.best_pred)
        # print('check same dice loss weight:',x)

        # if epoch < 2:
        #     x = 0
        # elif epoch <5 :
        #     x = 0.5
        # elif epoch < 10:
        #     x = 1
        # elif epoch < 15:
        #     x = 1.5
        # elif epoch < 20:
        #     x = 2
        # else:
        #     x = 3

        # todo 要让wce loss和dice loss的加权之和为1
        if epoch < 2:
            x = 0
        elif epoch < 5:
            x = 0.25
        elif epoch < 10:
            x = 0.5
        elif epoch < 15:
            x = 0.6
        elif epoch < 20:
            x = 0.7
        else:
            x = 0.8

        # validation 的时候没有loss.backward。那我就没有必要搞这些啊……

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            # print('output:', np.unique(output))
            # loss_ce,loss_dice = self.criterion(output, target)
            loss_ce, class_dice = self.criterion(output, target)
            # avg_dice_loss = (1 - (sum(class_dice) / len(class_dice))) / self.batch_size
            # print('avg dice loss',avg_dice_loss,'\n')
            # todo 根据class dice 和balance miou weight计算dice loss
            balance_dice_loss = [class_dice[i] * self.balance_m_weights[i] for i in range(len(class_dice))]
            loss_dice = (1 - sum(balance_dice_loss)) / self.batch_size
            # print('ce loss', loss_ce, '\n')
            # print('dice loss', loss_dice)

            loss = (1 - x) * loss_ce   + x * loss_dice
            test_loss += loss.item()
            test_ce_loss += loss_ce.item()
            test_dice_loss += loss_dice.item()
            tbar.set_description('Test loss: %.4f' % (test_loss / (i + 1)))  # set description是显示进度条的
            pred = output.data.cpu().numpy()

            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # print('predddd',np.unique(pred))  # 这里是4个类别了
            # Add batch sample into evaluator
            # print('target shape',target.shape)
            # print('pred shape',pred.shape)
            self.evaluator.add_batch(target, pred)   # 所有batch的一起计算acc，miou等

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU,class_miou = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/total_ce_loss_epoch', test_ce_loss, epoch)
        self.writer.add_scalar('val/total_dice_loss_epoch', test_dice_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.5f,CE Loss : %5f,Dice Loss : %5f' % (test_loss,test_ce_loss,test_dice_loss))

        self.balance_m_weights = self.cal_weights(mious=class_miou[1:])   #更新balance miou weights
        print('epoch {} miou balanced weight : {}'.format(epoch,self.balance_m_weights))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True             # 如果是最优的
            self.best_pred = new_pred   #则要更新best pred
        else:
            is_best = False

        self.saver.save_checkpoint({  #每一次都要保存模型，并且看是否是best更新model best
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)


        return self.best_pred



def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',       #
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,      # 这里的是指下采样的倍数，为2^4，从512下采样4次得到32.
                        help='network output stride (default: 8)')  #而秋茄总pixel为454个，还是两个小小的区块，每一个区块占据不过是20*20,20下采样4次成了1，会丢失了该信息了
    parser.add_argument('--dataset', type=str, default='mangrove',
                        choices=['pascal', 'coco', 'cityscapes', 'mangrove'],  # 加上mangrove
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='multi',
                        choices=['ce', 'focal','multi'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,     # 注意注意！！
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.007, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='step2',
                        choices=['poly', 'step', 'cos', 'step2'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str,default=None,
                        # default='/mnt/project/deeplab_mobilenet_npz+kset/run/mangrove/deeplab-resnet_5set_cut10/model_best.pth.tar',
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

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

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'mangrove': 1000,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'mangrove': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'wce+add_dice-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)

    # todo 统计所有数据的数量
    base_dir = Path.db_root_dir(dataset='mangrove')
    print('train data :',base_dir)
    base_dir = os.path.join(base_dir, 'JPEGImages')
    total_number = len(os.listdir(base_dir))
    total_index = [i for i in range(total_number)]
    random.shuffle(total_index)  # 打乱顺序

    trainer = Trainer(args.start_epoch, total_index, args)  # trainer生成log保存的路径，tensorboard，checkpoint等
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    # print('args.start epoch:',args.start_epoch)
    # print('Starting Epoch:', trainer.args.start_epoch)
    print('Starting Epoch:', args.start_epoch)
    # print('Total Epoches:', trainer.args.epochs)
    print('Total Epoches:', args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        #     如果我每次都要重新划分训练集和测试集的话，那我每个epoch都要重新做一次load train set和val set，在load的那边每次都打乱
        # trainer = Trainer(args)
        # todo 把数据分成k（10）份，每一份都有机会做训练集和验证集
        # todo 每个epoch都是训练10次验证10次，对10次的验证求平均作为该次训练的精度，再做模型的更新
        # todo 每5个epoch才对数据打乱一次
        trainer.train_loader, trainer.val_loader, trainer.test_loader, trainer.nclass = make_data_loader(epoch,
                                                                                                         total_index,
                                                                                                         args, **kwargs)
        # print('check index:',total_index[:10])
        # 并且要传入best_pred,因为每次调用Trainer，都会调用一次le_scheduler，会更新best_pred
        trainer.training(epoch)  # 训练
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):  # 测试
        trainer.validation(epoch)
        if epoch % 5 == 0:  # 10个epoch打乱index的顺序
            print('random')  # 打乱index
            random.shuffle(total_index)  # 每5次打乱一次顺序，不做训练，只做预测

    trainer.writer.close()


if __name__ == "__main__":
    main()
