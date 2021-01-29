import torch.backends.cudnn
import torch.cuda
import numpy as np
import random
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+123)
    np.random.seed(seed+1234)
    random.seed(seed+12345)
    torch.backends.cudnn.deterministic = True


def add_scalar(writer, user_num, test_result, epoch):
    test_loss, test_acc, user_loss, user_acc = test_result
    writer.add_scalar(f'user_{user_num}/global/test_loss', test_loss, epoch)
    writer.add_scalar(f'user_{user_num}/global/test_acc', test_acc, epoch)
    writer.add_scalar(f'user_{user_num}/local/test_loss', user_loss, epoch)
    writer.add_scalar(f'user_{user_num}/local/test_acc', user_acc, epoch)

