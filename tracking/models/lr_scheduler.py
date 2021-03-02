from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['LRScheduler', 'LogScheduler', 'StepScheduler', 'MultiStepScheduler', 'LinearStepScheduler', 'CosStepScheduler', 'WarmUPScheduler']

class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr
                for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__,
                                             self.lr_spaces)


