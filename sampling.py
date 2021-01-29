#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict
import random

def get_train_data(train_dataset,args):
    data_list_val = {}
    for j in range(args.num_classes):
        data_list_val[j] = [i for i, label in enumerate(train_dataset.targets) if label == j]
    idx_to_meta = []
    train_dict={i: np.array([], dtype=int) for i in range(args.num_classes)}

    for cls,img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        idx_to_meta.extend(img_id_list[:args.num_meta])
        train_dict[cls]=np.delete(img_id_list,np.arange(args.num_meta))

    random.shuffle(idx_to_meta)


    return train_dict,idx_to_meta





def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users




def data_noniid(train_dict,args):
    np.random.seed(666)
    random.seed(666)

    dict_users= {i: np.array([], dtype=int) for i in range(args.num_users)}
    class_size=len(train_dict[0])

    for n in range(args.num_classes):
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(args.num_users * [args.alpha]))
        for user in range(args.num_users):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list=train_dict[n][:min(len(train_dict[n]),num_imgs)]
            dict_users[user]=np.concatenate((dict_users[user],sampled_list),axis=0)
            train_dict[n]=train_dict[n][min(len(train_dict[n]), num_imgs):]
            random.shuffle(dict_users[user])
    return dict_users

