# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 03:04:48 2021

@author: krish
"""

import numpy as np
import pandas as pd
import scipy.io
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MplWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())

        self.toolbar = NavigationToolbar(self.canvas, self)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.toolbar)
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.axes.set_xlabel("Experiment")
        self.setLayout(vertical_layout)

def dist_calc(a, b):
    return math.sqrt((a[0] - b[0])**2+(a[1] - b[1])**2)
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(b[1]-a[1], b[0]-a[0]))
    if ang < -180:
        ang = 360 + ang
    elif ang > 180:
        ang = 360 - ang
    return ang

def angRect(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

def conv(path):
    extremes = [0 for i in range(4)]
    max_rad_dist = 0
    mat = scipy.io.loadmat(path)
    con_list = [element for element in mat]
    newData = [[i*0.0601431407 for i in range(len(mat['x_pos']))],
                       [item[0] for item in mat[con_list[7]]],
                       [item[0] for item in mat[con_list[8]]]]
    extremes = [min(newData[1]), min(newData[2]), max(newData[1]), max(newData[2])]
    newData = [[row[i] for row in newData] for i in range(len(newData[0]))]
    
    # columns = ['time_stamp', 'x_pos', 'y_pos', 'step_length', 'velocity', 'center_distance', 'angle']
    final_list = []
    for i in range(1, len(newData)):
        dist = dist_calc([newData[i][1], newData[i][2]], [newData[i-1][1], newData[i-1][2]])
        if dist < 5:
            final_list.append(newData[i])
    for i in range(1, len(final_list)):
        final_list[i].append(dist_calc([final_list[i-1][1], final_list[i-1][2]], [final_list[i][1], final_list[i][2]]))
        final_list[i].append(final_list[i][3]/0.0601431)
        final_list[i].append(dist_calc([final_list[0][1], final_list[0][2]], [final_list[i][1], final_list[i][2]]))
        if(final_list[i][5]>max_rad_dist):
            max_rad_dist = final_list[i][5]
        if(i!=1):
            final_list[i].append(getAngle((final_list[i-2][1], final_list[i-2][2]),
                                        (final_list[i-1][1], final_list[i-1][2]),
                                        (final_list[i][1], final_list[i][2])))
            
        # total_dist+=final_list[i][6]
    #     final_list[i].append(final_list[i][6]/0.0601431)
    #     total_speed+=final_list[i][7]
    
    # for i in range(2, len(final_list)):
    #     final_list[i].append(getAngle((final_list[i-2][4], final_list[i-2][5]), 
    #                                    (final_list[i-1][4], final_list[i-1][5]),
    #                                    (final_list[i][4], final_list[i][5])))
    #     if(final_list[i][8] < 20 and final_list[i][8] > -20):
    #         pause+=1
    

    # 
    # print(df)
    # return df, pause, total_dist/(len(newData)+1), total_speed/(len(newData)+1)
    return final_list, max_rad_dist, extremes

def calc(final_list, max_rad_dist, extremes):
    pause = 0
    total_dist = 0
    total_speed = 0
    linear_loco_count = 0
    bending_count = 0
    ang_pos = 0
    ang_pos_count = 0
    ang_neg = 0
    ang_neg_count = 0
    pos_lin_spread = 0
    neg_lin_spread = 0
    radial_spread = [0 for i in range(0, int(max_rad_dist), 10)]
    start_end_angle = angRect([final_list[0][1], final_list[0][2]], [final_list[-1][1], final_list[-1][2]])
    bbox_area = (extremes[2]-extremes[0]) * (extremes[3] - extremes[1])
    # print(bbox_area)
    for item in final_list:
        if(len(item)!=3):
            total_dist += item[3]
            total_speed += item[4]
            if(item[4] < 2.0):
                pause+=1
            for i in range(len(radial_spread)):
                if (item[5]>=i*10) and (item[5]<(i+1)*10):
                    radial_spread[i]+=1
                    break
        if(len(item)==7):
            if(item[6]>0):
                ang_pos_count+=1
                ang_pos+=item[6]
            if(item[6]<0):
                ang_neg+=item[6]
                ang_neg_count+=1
            if(abs(item[6])<20):
                linear_loco_count+=1
            if(abs(item[6]>45)):
                bending_count+=1
            if(angRect([final_list[0][1], final_list[0][2]], [item[1], item[2]]) > start_end_angle):
                pos_lin_spread += 1
            else:
                neg_lin_spread +=1
    # print(radial_spread)
    # print("pause {}, total_dist {}, total_speed {}, linear_Loco_count {} ".format(pause, total_dist, total_speed, linear_loco_count))
    return pause, total_dist/(len(final_list)-1), total_speed/(len(final_list)-1), linear_loco_count, bending_count, ang_pos/ang_pos_count, ang_neg/ang_neg_count, bbox_area, radial_spread, pos_lin_spread, neg_lin_spread
# df, pause, avg_dist, avg_speed = conv('mat_files/D3.mat')
# df, max_rad, bbox= conv('mat_files/D1.mat')
# calc(df, max_rad, bbox)
# print(pause)
# print(avg_dist)
# print(avg_speed)
# df.plot(x="x_pos", y="y_pos")

#0.0601431407
#Avg Linear Locomotion speed
#bounding box area
#major and minor angle
#radial scatter
#lateral scatter