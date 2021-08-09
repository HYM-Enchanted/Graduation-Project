import random

from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd,mangrove
from torch.utils.data import DataLoader
from mypath import Path
import os

def make_data_loader(epoch,index_list,args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        #if args.use_sbd:
         #   sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
          #  train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class


    # elif args.dataset == 'mangrove':
    #     # todo 在这里每次都重新划分训练集和测试集，训练集划分70%，测试集划分30%
    #     base_path = Path.db_root_dir(args.dataset)
    #     base_path = os.path.join(base_path,'JPEGImages')
    #     count_images = len(os.listdir(base_path))
    #     num = [i for i in range(count_images)]
    #     random.shuffle(num)   # 把index打乱，后面根据index读取数据   ,这样子保证每一次train 读入的数据都是全部的数据，并且split成train set和val set
    #
    #     train_set = mangrove.VOCSegmentation(args,index_list =num, split='train')   # 训练集
    #     val_set = mangrove.VOCSegmentation(args,index_list=num, split='val')   # 验证集
    #
    #     # test_set = mangrove.VOCSegmentation(args, split='test')  # 测试集（程序中没用上test set）
    #     # if args.use_sbd:
    #     #     sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
    #     #     train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
    #
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     # test_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=False, **kwargs)
    #
    #     return train_loader, val_loader, test_loader, num_class


    # elif args.dataset == 'mangrove':
    #     # todo 把所有的数据都load进来作为data_set
    #     # todo 在dataloader时再
    #
    #     data_set = mangrove.VOCSegmentation(args,split='train')   # load 所有的数据
    #     print('所有数据数量',len(data_set))
    #     k_num = len(data_set) // 10   # 每一份数据的数量（如果不能整除的话呢？）
    #     epoch_i = epoch % 10   # 说明是第几个epoch
    #     train_set = data_set[:epoch_i*k_num] + data_set[(epoch_i+1) * k_num:]
    #     val_set = data_set[epoch_i*k_num : (epoch_i+1) * k_num]
    #
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class
    #     # return data_set

    elif args.dataset == 'mangrove':
        k_num = len(index_list) // 5  # 每一份数据的数量（如果不能整除的话呢？）
        epoch_i = epoch % 5  # 说明是第几个epoch
        train_list = index_list[:k_num * epoch_i] + index_list[k_num *(epoch_i + 1):]
        train_set = mangrove.VOCSegmentation(args, index_list=train_list, split='train')  # 训练集

        val_list = index_list[k_num * epoch_i : k_num * (epoch_i + 1)]
        val_set = mangrove.VOCSegmentation(args,index_list = val_list, split='val')   # 验证集

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,drop_last=True, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

