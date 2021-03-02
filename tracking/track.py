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
    
    file_name = cfg["video"]
    write_video = True
    output_mp4 = file_name.split('.')[0]+'_tracked.mp4'
    frame_provider = VideoIterator(file_name)
    cx = 382.0
    cy = 321.0
    w = 30.0
    h = 30.0

    if cfg["output_video"]:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_mp4, fourcc, frame_provider.FPS,
            (im.shape[1], im.shape[0]), True)
        
    # tracking and visualization
    df = pd.DataFrame(columns=['time', 'X', 'Y'])
    temp = (int(cx), int(cy))
    frame_count = -1
    for im, t in frame_provider:
        frame_count += 1
        if frame_count == 0:
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            state = SiamRPN_init(im, target_pos, target_sz, model, 'SiamRPNPP')
            weight_img = np.zeros_like(im)
            df.loc[frame_count] = [t, cx + (w/2), cy + (h/2)]
            continue

        state = SiamRPN_track(state, im)
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        df.loc[frame_count] = [t, res[0] + res[2]/2, res[1] + res[3]/2]

        res = [int(l) for l in res]
        # print(res)
        cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
        center = (int(res[0] + res[2]/2), int(res[1] + res[3]/2))

        weight_img = cv2.line(weight_img, temp, center, (0, 0, 255), 5)
        #centers.append(center)
        #save_path = 'bio_frames_tracked/tracked_{}.png'.format(f)
        im = cv2.addWeighted(im, 1, weight_img, 1, 0)

        if cfg["output_video"]:
            writer.write(im)
        #cv2.imwrite(save_path, im)

        temp = center
    #print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
    writer.release() 
