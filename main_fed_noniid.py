#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from sampling import  data_noniid, cifar_iid, get_train_data
from options import args_parser
from Update import LocalUpdate,test_img,FedAvg
from util import setup_seed
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model import CNNCifar


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/fed/{}_{}_{}_C{}_iid{}_{}_user{}_{}'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                                                           args.alpha, args.num_users, current_time)
    # TAG = f'alpha_{alpha}/data_distribution'
    logdir = f'runs/{TAG}' if not args.debug else f'runs2/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)

    train_groups,idx_to_meta=get_train_data(train_dataset,args)

    if args.iid:
        dict_users = cifar_iid(train_dataset, args.num_users)
    else:
        dict_users = data_noniid(train_groups, args)


    # build model
    model = CNNCifar(args=args).to(args.device)

    print(model)
    model.train()

    # copy weights
    w_glob = model.state_dict()

    # training
    loss_train = []

    test_best_acc = 0.0

    for epoch in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(model).to(args.device))
            w_locals.append(w)
            loss_locals.append(loss)
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        model.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Train loss {:.3f}'.format(epoch+1, loss_avg))
        loss_train.append(loss_avg)
        writer.add_scalar('train_loss', loss_avg,epoch+1)
        test_acc, test_loss = test_img(model, test_dataset, args)
        writer.add_scalar('test_loss', test_loss, epoch+1)
        writer.add_scalar('test_acc', test_acc, epoch+1)

    # testing
    model.eval()
    acc_train, loss_train = test_img(model, train_dataset, args)
    acc_test, loss_test = test_img(model, test_dataset, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    writer.close()
