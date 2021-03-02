import torch
import torch.nn
import torch.nn.functional as F
import random
import numpy as np

def rpn_cross_entropy(input, target):
    r"""
    :param input: (15x15x5,2)
    :param target: (15x15x5,)
    :return:
    """
    mask_ignore = target == -1
    mask_calcu = 1 - mask_ignore
    loss = F.cross_entropy(input=input[mask_calcu], target=target[mask_calcu], size_average=False)
    # loss = torch.div(torch.sum(loss), 64)
    return loss
