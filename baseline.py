from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import os, sys
from bisect import bisect_right
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict

from reid.utils.data.sampler import RandomPairSampler
from reid.models.embedding2 import Sub_model
from reid.models.multi_branch import ENet
from reid.evaluators import Evaluator
from reid.trainers_partloss import Trainer
import pdb

import torch._utils 
try: 
    torch._utils._rebuild_tensor_v2 
except AttributeError: 
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride) 
        tensor.requires_grad = requires_grad 
        tensor._backward_hooks = backward_hooks 
        return tensor 
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def get_data(name, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval, np_ratio):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])
    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])


    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)
    train_loader = DataLoader(
        Preprocessor(list(set(train_set) | set(dataset.val)), root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,shuffle=True, pin_memory=True,drop_last=True)
        
    
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, val_loader,  test_loader

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    # print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)
    dataset, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 args.combine_trainval, args.np_ratio)

    # Create model
    base_model = models.create(args.arch, cut_at_pooling=True)    
    esub_model=Sub_model(num_features=256,in_features=2048, num_classes=751,FCN=True,dropout=0.5)#market1501 751,cuhk1367
    model = ENet(base_model, esub_model)
    
    model = nn.DataParallel(model).cuda()


    evaluator = Evaluator(model)


    # Load from checkpoint
    start_epoch = best_top1 = 0
    best_mAP = 0
    if args.resume:
#        checkpoint = load_checkpoint(args.resume)
#        if 'state_dict' in checkpoint.keys():
#            checkpoint = checkpoint['state_dict']
#        model.load_state_dict(checkpoint)
#        
#        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
#        model.module.load_state_dict(checkpoint['state_dict'])
        
        checkpoint = load_checkpoint(args.resume)
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
       
        
        print("Test the loaded model:")
        top1,mAP = evaluator.evaluate(test_loader, test_loader, dataset.query, dataset.gallery, dataset=args.dataset)

        best_mAP = mAP

    if args.evaluate:
        print("Test:")
        top1,mAP=evaluator.evaluate(test_loader, test_loader, dataset.query, dataset.gallery,dataset=args.dataset)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer
    param_groups = [
        {'params': model.module.base_model.parameters(), 'lr_mult': 1.0},
        {'params': model.module.embed_model.parameters(), 'lr_mult': 1.0}]
    optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay,nesterov=True)
    # Trainer
    #trainer = SiameseTrainer(model, criterion)
    trainer = Trainer(model, criterion, 0, 0, SMLoss_mode=0)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr * (0.1 ** (epoch // args.step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

#     Start training
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)#, base_lr=args.lr

        if epoch % args.eval_step==0:
            #pdb.set_trace()
            mAP = evaluator.evaluate(val_loader, val_loader,dataset.val, dataset.val, top1=False)#, top1=False
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch':epoch +1,
                'best_top1':best_top1,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader,test_loader,dataset.query, dataset.gallery,dataset=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Siamese reID baseline")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,default=384,
                        help="input height, default: 256 for resnet")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--np-ratio', type=int, default=3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--step-size', type=int, default=40)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=20, help="evaluation step")
    parser.add_argument('--seed', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'datasets'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'checkpoints'))
    main(parser.parse_args())
