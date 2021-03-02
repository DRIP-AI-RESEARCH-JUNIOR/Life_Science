import argparse
import numpy as np
import time
import pandas as pd
import cv2

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



class VideoIterator(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.FPS = cv2.VideoCapture(file_name).get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        self.total_time = 0
        self.cap = cv2.VideoCapture(self.file_name)
        #self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        elapsed = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        #self.total_time += elapsed
        
        if not was_read:
            self.cap.release()
            raise StopIteration
        return img, elapsed
    
if __name__=="__main__":
    
    model = SiamRPNPPRes50(cfg['model'])
    load_net(cfg['weight'], model)
    model.eval().cuda()
    
    
