# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'final.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QGuiApplication
import cv2
import sys, os

from tracking.utils import VideoIterator
from tracking.track_util import process_track, first_frame

cfg = {
    "model": "SiamRPNPP",
    "weight": os.path.join(os.getcwd(), "weight\\SiamRPNPPRes50_cpu.pth"),
    "output_video": True
}

class MyLabel(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    #

    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()
        #

    def mouseReleaseEvent(self, event):
        self.flag = False     
        cfg["cx"], cfg["cy"], cfg["w"], cfg["h"] = self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0)
        # print(cfg)
        #

    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()
         #  event

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QRect(self.x0, self.y0, abs(
            self.x1-self.x0), abs(self.y1-self.y0))
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
        painter.drawRect(rect)


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1051, 827)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_video = QtWidgets.QLabel(self.centralwidget)
        self.label_video.setGeometry(QtCore.QRect(20, 20, 161, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_video.sizePolicy().hasHeightForWidth())
        self.label_video.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("OCR A Std")
        font.setPointSize(11)
        self.label_video.setFont(font)
        self.label_video.setObjectName("label_video")
        self.textBrowser_video = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_video.setGeometry(QtCore.QRect(195, 15, 681, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_video.sizePolicy().hasHeightForWidth())
        self.textBrowser_video.setSizePolicy(sizePolicy)
        self.textBrowser_video.setObjectName("textBrowser_video")
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_video.setGeometry(QtCore.QRect(904, 15, 121, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Orator Std")
        font.setPointSize(12)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.label_output = QtWidgets.QLabel(self.centralwidget)
        self.label_output.setGeometry(QtCore.QRect(20, 82, 161, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_output.sizePolicy().hasHeightForWidth())
        self.label_output.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("OCR A Std")
        font.setPointSize(11)
        self.label_output.setFont(font)
        self.label_output.setObjectName("label_output")
        self.pushButton_output = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_output.setGeometry(QtCore.QRect(904, 77, 121, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_output.sizePolicy().hasHeightForWidth())
        self.pushButton_output.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Orator Std")
        font.setPointSize(12)
        self.pushButton_output.setFont(font)
        self.pushButton_output.setObjectName("pushButton_output")
        self.textBrowser_output = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_output.setGeometry(QtCore.QRect(195, 77, 681, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_output.sizePolicy().hasHeightForWidth())
        self.textBrowser_output.setSizePolicy(sizePolicy)
        self.textBrowser_output.setObjectName("textBrowser_output")
        self.pushButton_process = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_process.setGeometry(QtCore.QRect(470, 730, 121, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_process.sizePolicy().hasHeightForWidth())
        self.pushButton_process.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Orator Std")
        font.setPointSize(12)
        self.pushButton_process.setFont(font)
        self.pushButton_process.setObjectName("pushButton_process")
        self.groupBox_frame = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_frame.setGeometry(QtCore.QRect(20, 140, 1011, 571))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_frame.sizePolicy().hasHeightForWidth())
        self.groupBox_frame.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("OCR A Std")
        font.setPointSize(11)
        self.groupBox_frame.setFont(font)
        self.groupBox_frame.setObjectName("groupBox_frame")
        self.progress = QtWidgets.QLabel(self.centralwidget)
        self.progress.setGeometry(QtCore.QRect(22, 745, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(11)
        self.progress.setFont(font)
        self.progress.setText("")
        self.progress.setObjectName("progress")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1051, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton_video.clicked.connect(self.getFile)
        self.pushButton_output.clicked.connect(self.getOutput)
        self.pushButton_process.clicked.connect(self.final_process)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_video.setText(_translate("MainWindow", "Video File :"))
        self.pushButton_video.setText(_translate("MainWindow", "UPLOAD"))
        self.label_output.setText(_translate("MainWindow", "Output :"))
        self.pushButton_output.setText(_translate("MainWindow", "SELECT"))
        self.pushButton_process.setText(_translate("MainWindow", "PROCESS"))
        self.groupBox_frame.setTitle(_translate("MainWindow", "First Frame"))

    def getFile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Open files', os.path.abspath(__file__))[0]
        if fname:
            cfg['video'] = fname
            self.textBrowser_video.setText(fname)
            self.centralwidget.lb = MyLabel(self.centralwidget)
            self.centralwidget.lb.setGeometry(QRect(35, 165, 980, 530))
            self.img = first_frame(fname)
            height, width, bytesPerComponent = self.img.shape
            self.factor = width/945
            bytesPerLine = 3 * width
            cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
            QImg = QImage(self.img.data, width, height,
                          bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg)
            pixmap = pixmap.scaled(945, 945, QtCore.Qt.KeepAspectRatio)
            self.centralwidget.lb.setPixmap(pixmap)
            self.centralwidget.lb.setCursor(Qt.CrossCursor)         
            self.centralwidget.lb.show()

    def getOutput(self):
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None, caption='Choose Directory', directory=os.getcwd())
        if selected_dir:
            cfg["output_path"] = selected_dir
            self.textBrowser_output.setText(selected_dir)

    def final_process(self):
        if "cx" in cfg.keys() and "output_path" in cfg.keys():
            # print("Factor", self.factor)
            cfg["cx"], cfg["cy"], cfg["w"], cfg["h"] = int(
                cfg["cx"]*self.factor), int(cfg["cy"]*self.factor), int(cfg["w"]*self.factor), int(cfg["h"]*self.factor)
            process_track(cfg)
            self.progress.setText("Processed")
        else:
            if "cx" not in cfg.keys():
                self.progress.setText("Draw Bounding Box")
            elif "output_path" not in cfg.keys():
                self.progress.setText("Select output path")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
