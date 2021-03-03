import argparse
import numpy as np
import time, os, cv2
import pandas as pd

import torch
from torch import nn

from tracking.models.builder import SiamRPNPP
from tracking.runRPN import SiamRPN_init, SiamRPN_track
from tracking.utils import load_net, VideoIterator, cxy_wh_2_rect

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
    def __init__(self, tracker_name="SiamRPNPP"):
        super(SiamRPNPPRes50, self).__init__(tracker_name)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 255, 'adaptive': False} # 0.355


def first_frame(fname):
    f_provider = VideoIterator(fname)
    i = iter(f_provider)
    img, elapsed = next(i)
    return img
    
def process_track(cfg):
    
    model = SiamRPNPPRes50(cfg['model'])
    load_net(cfg['weight'], model)
    model.eval().cuda()
    
    file_name = cfg["video"]
    write_video = True
    output_mp4 = os.path.join(cfg["output_path"], os.path.basename(file_name).split('.')[0]+'_tracked.mp4')
    # print("Output mp4 file : {}".format(output_mp4))
    frame_provider = VideoIterator(file_name)
    cx = cfg["cx"]
    cy = cfg["cy"]
    w = cfg["w"]
    h = cfg["h"]
        
    # tracking and visualization
    df = pd.DataFrame(columns=['Time', 'CX', 'CY', 'W', 'H', 'Score'])
    temp = (int(cx), int(cy))
    frame_count = -1
    toc = 0
    for im, t in frame_provider:
        tic = cv2.getTickCount()
        frame_count += 1
        if frame_count == 0:
            
            if cfg["output_video"]:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_mp4, fourcc, frame_provider.FPS,
                    (im.shape[1], im.shape[0]), True)
            
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            state = SiamRPN_init(im, target_pos, target_sz, model, cfg["model"])
            toc += cv2.getTickCount()-tic
            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            weight_img = np.zeros_like(im)
            df.loc[frame_count] = [t, *res, 1]
            continue

        state = SiamRPN_track(state, im)
        toc += cv2.getTickCount()-tic
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        df.loc[frame_count] = [t, *res, state['score']]

        res = [int(l) for l in res]
        
        center = (int(res[0] + res[2]/2), int(res[1] + res[3]/2))        

        if cfg["output_video"]:
            cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
            weight_img = cv2.line(weight_img, temp, center, (0, 0, 255), 5)
            im = cv2.addWeighted(im, 1, weight_img, 1, 0)
            
            writer.write(im)

        temp = center

    writer.release() 
    
    print('Tracking Speed {:.1f}fps'.format(frame_count/(toc/cv2.getTickFrequency())))
    
    save_csv = os.path.join(cfg["output_path"], os.path.basename(file_name).split('.')[0] + '.csv')
    # print("CSV file path {}".format(save_csv))
    df.to_csv(save_csv)
