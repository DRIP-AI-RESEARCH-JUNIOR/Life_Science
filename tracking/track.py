import argparse
import numpy as np
import time
import pandas as pd

import torch
from torch import nn

from models.builder import SiamRPNPP
from runRPN import SiamRPN_init, SiamRPN_track
from utils import load_net

cfg = {
    "model": "SiamRPNPP",
    "video": "abc.mp4",
    "weight": "pqr.pth",
    "output_video": True
}

class SiamRPNPP_N(nn.Module):
    def __init__(self, tracker_name):
        super(SiamRPNPP_N, self).__init__()
        self.tracker_name = tracker_name
        self.model = SiamRPNPP()

    def temple(self, z):
        zf = self.model.features(z)
        zf = self.model.neck(zf)
        self.zf = zf

    def forward(self, x):
        xf = self.model.features(x)
        xf = self.model.neck(xf)
        cls, loc = self.model.head(self.zf, xf)
        return loc, cls


class SiamRPNPPRes50(SiamRPNPP_N):
    def __init__(self, tracker_name='SiamRPNPP'):
        super(SiamRPNPPRes50, self).__init__(tracker_name)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 255, 'adaptive': False} # 0.355

