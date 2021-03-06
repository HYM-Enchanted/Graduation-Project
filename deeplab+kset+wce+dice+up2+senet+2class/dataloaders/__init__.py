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
    #     # todo ?????????????????????????????????????????????????????????????????????70%??????????????????30%
    #     base_path = Path.db_root_dir(args.dataset)
    #     base_path = os.path.join(base_path,'JPEGImages')
    #     count_images = len(os.listdir(base_path))
    #     num = [i for i in range(count_images)]
    #     random.shuffle(num)   # ???index?????????????????????index????????????   ,????????????????????????train ?????????????????????????????????????????????split???train set???val set
    #
    #     train_set = mangrove.VOCSegmentation(args,index_list =num, split='train')   # ?????????
    #     val_set = mangrove.VOCSegmentation(args,index_list=num, split='val')   # ?????????
    #
    #     # test_set = mangrove.VOCSegmentation(args, split='test')  # ??????????????????????????????test set???
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
    #     # todo ?????????????????????load????????????data_set
    #     # todo ???dataloader??????
    #
    #     data_set = mangrove.VOCSegmentation(args,split='train')   # load ???????????????
    #     print('??????????????????',len(data_set))
    #     k_num = len(data_set) // 10   # ????????????????????????????????????????????????????????????
    #     epoch_i = epoch % 10   # ??????????????????epoch
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
        k_num = len(index_list) // 5  # ????????????????????????????????????????????????????????????
        epoch_i = epoch % 5  # ??????????????????epoch
        train_list = index_list[:k_num * epoch_i] + index_list[k_num *(epoch_i + 1):]
        train_set = mangrove.VOCSegmentation(args, index_list=train_list, split='train')  # ?????????

        val_list = index_list[k_num * epoch_i : k_num * (epoch_i + 1)]
        val_set = mangrove.VOCSegmentation(args,index_list = val_list, split='val')   # ?????????

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,**kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True,**kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

