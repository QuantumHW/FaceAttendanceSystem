import os
import cv2
import sys
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from menu import Ui_menuWindow
from register import Ui_registerWindow
from attendance import Ui_attendanceWindow
import numpy as np
import dlib
import csv
from tools import *
_translate = QtCore.QCoreApplication.translate
cap = cv2.VideoCapture(0)  # 全局摄像头
count = 0  # 计数
# 开始注册时间
startTime = time.time()
# 视频时间
frameTime = startTime
# 控制显示打卡成功的时长
show_time = (startTime - 10)
# 界面缓存
ui_register = None
ui_attendance = None
# 缓存用户信息
user_id = None
user_name = None
# 加载人脸检测器
hog_face_detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')
haar_face_detector = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

# 加载关键点检测器
points_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
# 加载resnet模型
face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')





def register_start(self):
    # 开始录入人脸
    global count, startTime, frameTime, show_time, cap, ui_register, user_id, user_name
    # 初始化全局参数
    cap.release()
    timer_register_show_pic.stop()
    ui_register.label_register_show.setText(_translate("MainWindow", "暂无数据输入"))
    count = 0
    startTime = time.time()
    frameTime = startTime
    show_time = (startTime - 10)
    try:
        text, ok_is_pressed = QInputDialog.getText(None,"请输入姓名", "姓名:", QLineEdit.Normal, "")
        if ok_is_pressed and text != '':
            user_name = text
            text, ok_is_pressed = QInputDialog.getText(None,"请输入工号", "工号:",  QLineEdit.Normal, "")
            if ok_is_pressed and text != '':
                user_id = text
                cap = cv2.VideoCapture(0)
                timer_register_show_pic.start(5)
    except Exception as e:
        print(str(e))


def register_show_pic():
    global count, startTime, frameTime, show_time, ui_register, cap, user_id, user_name
    interval = 3
    faceCount = 3
    try:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # 检测
            with open('./data/face_feature.csv', 'a', newline='') as f:
                face_detection = hog_face_detector(frame, 1)
                for face in face_detection:
                    # 识别68个关键点
                    points = points_detector(frame, face)
                    # 绘制人脸关键点
                    for point in points.parts():
                        cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), 1)
                    # 绘制框框
                    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
                    cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                    now = time.time()
                    # 检查次数
                    if count < faceCount:
                        # 检查时间
                        if now - startTime > interval:
                            # 特征描述符
                            face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)
                            face_descriptor = [f for f in face_descriptor]
                            # 描述符增加进data文件
                            face_data_temp = [user_id, user_name, face_descriptor]
                            # with open('./data/feature.csv', 'a', newline='') as f:
                            csv_writer = csv.writer(f)
                            # 写入
                            csv_writer.writerow(face_data_temp)
                            register_out_text(
                                '人脸注册成功 {count}/{faceCount}，用户ID:{faceId}，用户姓名:{userName}'.format(
                                    count=(count + 1),
                                    faceCount=faceCount,
                                    faceId=user_id,
                                    userName=user_name))
                            sho_time = time.time()
                            # 时间重置
                            startTime = now
                            # 次数加一
                            count += 1
                    else:
                        register_out_text('已完成注册，请返回主界面')
                        timer_register_show_pic.stop()
                        cap.release()
            cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width / ui_register.label_register_show.width(),
                        height / ui_register.label_register_show.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            ui_register.label_register_show.setAlignment(Qt.AlignCenter)
            ui_register.label_register_show.setPixmap(pixmap)
    except Exception as e:
        register_out_text(str(e))


def register_out_text(text):
    global ui_register
    timestamp = '[' + time.strftime("%Y/%m/%d-%H:%M") + ']'
    ui_register.text_register_out.append(timestamp + text)


def register_cancel():
    # 取消（返回主界面）
    global cap
    timer_register_show_pic.stop()
    cap.release()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.button_register.clicked.connect(open_register)
    ui.button_attendance.clicked.connect(open_attendance)
    ui.button_management.clicked.connect(open_management)


def attendance_show_pic():

    pass


def attendance_start():
    # 考勤打卡
    pass


def attendance_back():
    # 返回主界面
    global cap
    timer_attendance_show_pic.stop()
    cap.release()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.button_register.clicked.connect(open_register)
    ui.button_attendance.clicked.connect(open_attendance)
    ui.button_management.clicked.connect(open_management)

def open_register():
    # 打开注册界面
    global ui_register
    ui_register = Ui_registerWindow()
    ui_register.setupUi(MainWindow)
    MainWindow.show()

    ui_register.button_register_start.clicked.connect(register_start)
    ui_register.button_register_cancel.clicked.connect(register_cancel)


def open_attendance():
    # 打开考勤界面
    global ui_attendance
    ui_attendance = Ui_attendanceWindow()
    ui_attendance.setupUi(MainWindow)
    MainWindow.show()

    ui_attendance.button_attendance_start.clicked.connect(attendance_start)
    ui_attendance.button_attendance_back.clicked.connect(attendance_back)



def open_management():
    # 打开信息管理界面
    pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_menuWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()  # 显示主界面

    ui.button_register.clicked.connect(open_register)
    ui.button_attendance.clicked.connect(open_attendance)
    ui.button_management.clicked.connect(open_management)

    timer_register_show_pic = QTimer()
    timer_register_show_pic.timeout.connect(register_show_pic)

    timer_attendance_show_pic = QTimer()
    timer_attendance_show_pic.timeout.connect(attendance_show_pic)
    sys.exit(app.exec_())
