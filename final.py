# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test7.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import numpy as np
import os
from backup_1 import conv, calc, MplWidget
base_dir = os.getcwd()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1115, 912)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setMinimumSize(QtCore.QSize(0, 40))
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 0, 1, 3)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1070, 3235))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.label_Trajectory = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_Trajectory.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Trajectory.setFont(font)
        self.label_Trajectory.setObjectName("label_Trajectory")
        self.verticalLayout_2.addWidget(self.label_Trajectory)
        self.trajectory = MplWidget(self.scrollAreaWidgetContents)
        self.trajectory.setMinimumSize(QtCore.QSize(0, 800))
        self.trajectory.setObjectName("bbox")
        self.verticalLayout_2.addWidget(self.trajectory)

        self.label_speed = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_speed.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_speed.setFont(font)
        self.label_speed.setObjectName("label_speed")
        self.verticalLayout_2.addWidget(self.label_speed)
        self.avgSpeed = MplWidget(self.scrollAreaWidgetContents)
        self.avgSpeed.setMinimumSize(QtCore.QSize(0, 800))
        self.avgSpeed.setObjectName("avgSpeed")
        self.verticalLayout_2.addWidget(self.avgSpeed)
        self.label_locomtion = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_locomtion.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_locomtion.setFont(font)
        self.label_locomtion.setObjectName("label_locomtion")
        self.verticalLayout_2.addWidget(self.label_locomtion)
        self.avgLocomotion = MplWidget(self.scrollAreaWidgetContents)
        self.avgLocomotion.setMinimumSize(QtCore.QSize(0, 800))
        self.avgLocomotion.setObjectName("avgLocomotion")
        self.verticalLayout_2.addWidget(self.avgLocomotion)
        self.label_bending = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_bending.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_bending.setFont(font)
        self.label_bending.setObjectName("label_bending")
        self.verticalLayout_2.addWidget(self.label_bending)
        self.avgBending = MplWidget(self.scrollAreaWidgetContents)
        self.avgBending.setMinimumSize(QtCore.QSize(0, 800))
        self.avgBending.setObjectName("avgBending")
        self.verticalLayout_2.addWidget(self.avgBending)
        self.label_pause = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_pause.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_pause.setFont(font)
        self.label_pause.setObjectName("label_pause")
        self.verticalLayout_2.addWidget(self.label_pause)
        self.pause = MplWidget(self.scrollAreaWidgetContents)
        self.pause.setMinimumSize(QtCore.QSize(0, 800))
        self.pause.setObjectName("pause")
        self.verticalLayout_2.addWidget(self.pause)
        self.label_linear = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_linear.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_linear.setFont(font)
        self.label_linear.setObjectName("label_linear")
        self.verticalLayout_2.addWidget(self.label_linear)
        self.linearLoco = MplWidget(self.scrollAreaWidgetContents)
        self.linearLoco.setMinimumSize(QtCore.QSize(0, 800))
        self.linearLoco.setObjectName("avgAngle")
        self.verticalLayout_2.addWidget(self.linearLoco)
        self.label_angle = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_angle.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_angle.setFont(font)
        self.label_angle.setObjectName("label_pause")
        self.verticalLayout_2.addWidget(self.label_angle)
        self.angle = MplWidget(self.scrollAreaWidgetContents)
        self.angle.setMinimumSize(QtCore.QSize(0, 800))
        self.angle.setObjectName("pause")
        self.verticalLayout_2.addWidget(self.angle)
        self.label_bbox = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_bbox.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_bbox.setFont(font)
        self.label_bbox.setObjectName("label_bbox")
        self.verticalLayout_2.addWidget(self.label_bbox)
        self.bbox = MplWidget(self.scrollAreaWidgetContents)
        self.bbox.setMinimumSize(QtCore.QSize(0, 800))
        self.bbox.setObjectName("bbox")
        self.verticalLayout_2.addWidget(self.bbox)

        self.label_linSpread = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_linSpread.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_linSpread.setFont(font)
        self.label_linSpread.setObjectName("label_linSpread")
        self.verticalLayout_2.addWidget(self.label_linSpread)
        self.linSpread = MplWidget(self.scrollAreaWidgetContents)
        self.linSpread.setMinimumSize(QtCore.QSize(0, 800))
        self.linSpread.setObjectName("bbox")
        self.verticalLayout_2.addWidget(self.linSpread)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 3, 0, 1, 3)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setFamily("Myriad Pro Cond")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 1, 0, 1, 1)
        self.process = QtWidgets.QPushButton(self.centralwidget)
        self.process.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setFamily("Myriad Pro Cond")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.process.setFont(font)
        self.process.setObjectName("process")
        self.gridLayout.addWidget(self.process, 1, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1115, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.buttonClick)
        self.process.clicked.connect(self.plot)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowIcon(QtGui.QIcon('images/drip.ico'))
        MainWindow.setWindowTitle(_translate("MainWindow", "DRIP"))
        self.label_Trajectory.setText(_translate("MainWindow", "Trajectory"))
        self.label_speed.setText(_translate("MainWindow", "Average Speed"))
        self.label_locomtion.setText(_translate("MainWindow", "Average Locomtion"))
        self.label_bending.setText(_translate("MainWindow", "Bending"))
        self.label_pause.setText(_translate("MainWindow", "Pause"))
        self.label_linear.setText(_translate("MainWindow", "Linear Locomotion"))
        self.label_angle.setText(_translate("MainWindow", "Average Angle"))
        self.label_bbox.setText(_translate("MainWindow", "Bounding Box Area"))
        self.label_linSpread.setText(_translate("MainWindow", "Linear Spread"))
        self.pushButton.setText(_translate("MainWindow", "Upload"))
        self.process.setText(_translate("MainWindow", "Plot"))

    def trajectory_map(self, fig, x_list, y_list, expNo):
        fig.canvas.axes.clear()
        for xitem, yitem in zip(x_list, y_list):
            fig.canvas.axes.plot(xitem, yitem)
        fig.canvas.axes.legend(expNo)
        fig.canvas.draw()

    def update_graph(self, fig, x, y, z):
        fig.canvas.axes.clear()
        if z==None:
            fig.canvas.axes.bar(x, y)
            for p in fig.canvas.axes.patches:
                fig.canvas.axes.annotate(format(p.get_height(), '.001f'),
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center',
                               xytext=(0, 4),
                               textcoords='offset points')
        else:
            barWidth = 0.25
            br1 = np.arange(len(x))
            br2 = [x + barWidth for x in br1]
            fig.canvas.axes.bar(br1, y, width = barWidth, label="Positive")
            fig.canvas.axes.bar(br2, z, width = barWidth, label="Negative")
            for p in fig.canvas.axes.patches:
                fig.canvas.axes.annotate(format(p.get_height(), '.001f'),
                                         (p.get_x() + p.get_width() / 2., p.get_height()),
                                         ha='center', va='center',
                                         xytext=(0, 4),
                                         textcoords='offset points')
            fig.canvas.axes.legend(('Positive', 'Negative'))
            fig.canvas.axes.set_xticks([r + barWidth/2 for r in range(len(x))])
            fig.canvas.axes.set_xticklabels(x)
        fig.canvas.axes.set_xlabel("Experiment")
        fig.canvas.draw()

    def plot(self):
        self.trajectory_map(self.trajectory, self.xList, self.yList, self.expNo)
        self.update_graph(self.avgSpeed, self.expNo, self.avgSpeedVal, None)
        self.update_graph(self.avgLocomotion, self.expNo, self.avgDistVal, None)
        self.update_graph(self.pause, self.expNo, self.pauseCount, None)
        self.update_graph(self.avgBending, self.expNo, self.bendCount, None)
        self.update_graph(self.linearLoco, self.expNo, self.linearLocoTime, None)
        self.update_graph(self.angle, self.expNo, self.posAngle, self.negAngle)
        self.update_graph(self.bbox,self.expNo, self.bbox_area, None)
        self.update_graph(self.linSpread, self.expNo, self.pos_lin_spread, self.neg_lin_spread)

    def getFile(self):
        fname = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open files', base_dir, "Mat files(*.mat)")[0]
        self.lineEdit.setText(", ".join(str(x) for x in fname))
        self.avgSpeedVal = []
        self.avgDistVal = []
        self.expNo = []
        self.linearLocoTime = []
        self.pauseCount = []
        self.bendCount = []
        self.posAngle = []
        self.negAngle = []
        self.bbox_area = []
        self.rad_spread = []
        self.pos_lin_spread = []
        self.neg_lin_spread = []
        self.xList = []
        self.yList = []
        for item in fname:
            self.expNo.append(item.split("/")[-1].replace(".mat", ""))
            final_list, max_rad, bbox = conv(item)
            pause_count, avg_dist, avg_speed, linear_count, bend_count, avg_pos_angle, avg_neg_angle, bbox_area, rad_spread_count, pos_lin_count, neg_lin_count = calc(final_list, max_rad, bbox)
            self.xList.append([coord[1] for coord in final_list])
            self.yList.append([coord[2] for coord in final_list])
            self.avgSpeedVal.append(avg_speed)
            self.avgDistVal.append(avg_dist)
            self.pauseCount.append(pause_count)
            self.linearLocoTime.append(linear_count)
            self.bendCount.append(bend_count)
            self.posAngle.append(avg_pos_angle)
            self.negAngle.append(-1*avg_neg_angle)
            self.bbox_area.append(bbox_area)
            self.rad_spread.append(rad_spread_count)
            self.pos_lin_spread.append(pos_lin_count)
            self.neg_lin_spread.append(neg_lin_count)

    def buttonClick(self):
        self.getFile()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
