import torch
from torch import nn

import numpy as np

def center_crop(x):
    """
    center crop layer. crop [1:-2] to eliminate padding influence.
    Crop 1 element around the tensor
    input x can be a Variable or Tensor
    """
    return x[:, :, 1:-1, 1:-1].contiguous()


def center_crop7(x):
    """
    Center crop layer for stage1 of resnet. (7*7)
    input x can be a Variable or Tensor
    """

    return x[:, :, 2:-2, 2:-2].contiguous()
