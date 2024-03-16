import cv2
import sys
import time
import dlib
from tools import *
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from menu import Ui_menuWindow
from register import Ui_registerWindow
from attendance import Ui_attendanceWindow
from management import Ui_managementWindow

_translate = QtCore.QCoreApplication.translate
cap = cv2.VideoCapture(0)  # 全局摄像头
count = 0  # 计数
face_count = 0
# 开始注册时间
startTime = time.time()
# 视频时间
frameTime = startTime
# 控制显示打卡成功的时长
show_time = (startTime - 10)
# 界面缓存
ui = None
# 缓存用户信息
user_id = None
user_name = None
feature_list = None
label_list = None
name_list = None
# 缓存文件标识符
opened_file = None
csv_writer = None
# 加载人脸检测器
hog_face_detector = dlib.get_frontal_face_detector()
haar_face_detector = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')
# 加载关键点检测器
points_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
# 加载resnet模型
face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')


def register_start(self):
    # 开始录入人脸
    global count, startTime, frameTime, show_time, cap, ui, user_id, user_name, opened_file, csv_writer
    # 初始化全局参数
    cap.release()
    timer_register_show_pic.stop()
    ui.label_register_show.setText(_translate("MainWindow", "暂无数据输入"))
    count = 0
    startTime = time.time()
    frameTime = startTime
    show_time = (startTime - 10)
    try:
        text, ok_is_pressed = QInputDialog.getText(None,"请输入姓名", "请准确输入您的姓名:", QLineEdit.Normal, "")
        if ok_is_pressed and text != '':
            user_name = text
            text, ok_is_pressed = QInputDialog.getText(None,"请输入工号", "请准确输入您的工号:",  QLineEdit.Normal, "")
            if ok_is_pressed and text != '':
                user_id = text
                opened_file = open('./data/face_feature.csv', 'a+', newline='')
                csv_writer = csv.writer(opened_file)
                cap = cv2.VideoCapture(0)
                timer_register_show_pic.start(5)
    except Exception as e:
        print(str(e))


def register_show_pic():
    global count, startTime, frameTime, show_time, ui, cap, user_id, user_name, opened_file, csv_writer
    interval = 3
    faceCount = 3
    try:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # 检测
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
                        # 写入
                        csv_writer.writerow(face_data_temp)
                        register_out_text(
                            '人脸注册进度:{count}/{faceCount}，用户ID:{faceId}，用户姓名:{userName}'.format(
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
                    QMessageBox.information(None, "完成", '恭喜！用户' + str(user_name) + "注册成功,请返回主界面",
                                            QMessageBox.Yes, QMessageBox.Yes)
                    timer_register_show_pic.stop()
                    opened_file.close()
                    cap.release()
            cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            ratio = max(width / ui.label_register_show.width(),
                        height / ui.label_register_show.height())
            pixmap.setDevicePixelRatio(ratio)
            ui.label_register_show.setAlignment(Qt.AlignCenter)
            ui.label_register_show.setPixmap(pixmap)
    except Exception as e:
        register_out_text(str(e))


def register_out_text(text):
    global ui
    timestamp = '[' + time.strftime("%Y/%m/%d-%H:%M") + ']'
    ui.text_register_out.append(timestamp + text)


def show_camera():
    # 显示摄像头画面（不做任何处理）
    global cap, ui
    try:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width / ui.label_attendance_show.width(),
                        height / ui.label_attendance_show.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            ui.label_attendance_show.setAlignment(Qt.AlignCenter)
            ui.label_attendance_show.setPixmap(pixmap)
    except Exception as e:
        print('摄像头异常:'+str(e))


def attendance_start():  # 考勤打卡
    global cap, feature_list, label_list, name_list, ui
    threshold = 0.5
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # 人脸检测
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detection = haar_face_detector.detectMultiScale(frame_gray, minNeighbors=7, minSize=(100, 100))
    predict_name = "FAILED"
    face_time = time.strftime("%Y/%m/%d-%H:%M:%S")
    ui.label_time.setText(_translate("attendanceWindow", "时间：" + str(face_time)))
    for face in face_detection:
        l, t, r, b = get_dlib_rect(face)
        face = dlib.rectangle(l, t, r, b)
        points = points_detector(frame, face)  # 识别68个关键点
        face_crop = frame[t:b, l:r]  # 人脸区域
        # 特征
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        distance = np.linalg.norm((face_descriptor - feature_list), axis=1)  # 计算距离
        min_index = np.argmin(distance)  # 最小距离索引
        min_distance = distance[min_index]  # 最小距离
        if min_distance < threshold:  # 距离小于阈值，表示匹配
            predict_id = label_list[min_index]
            predict_name = name_list[min_index]
            ui.label_id.setText(_translate("attendanceWindow", "工号："+str(predict_id)))
            ui.label_name.setText(_translate("attendanceWindow", "姓名："+str(predict_name)))
            cur_frame = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, width*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            ratio = max(width / ui.label_photo.width(),
                        height / ui.label_photo.height())
            pixmap.setDevicePixelRatio(ratio)
            ui.label_photo.setAlignment(Qt.AlignCenter)
            ui.label_photo.setPixmap(pixmap)
            QMessageBox.information(None, "成功", str(predict_name) + "打卡成功！",
                                    QMessageBox.Yes, QMessageBox.Yes)
            line = [predict_id, predict_name, face_time]
            with open('./data/log_attendance.csv', 'a+', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(line)  # 写入考勤表
    if predict_name == "FAILED":
        ui.label_id.setText(_translate("attendanceWindow", "工号：未知"))
        ui.label_name.setText(_translate("attendanceWindow", "姓名：未知"))
        ui.label_photo.setText(_translate("attendanceWindow", "暂无人脸信息"))
        QMessageBox.warning(None, "打卡失败", "未识别到合法人脸！",
                            QMessageBox.Yes, QMessageBox.Yes)


def export_infor():
    # 导出考勤表
    filepath, type = QFileDialog.getSaveFileName(MainWindow, '导出考勤表',
                                                 '/records', 'Excel表格(*.xlsx)')
    if filepath:
        try:
            data_export = None
            count_data = 0
            with open('./data/log_attendance.csv', 'r') as f:
                csv_reader = csv.reader(f)
                rows_list = []  # 先使用列表，增加完数据后，再转回Dataframe，可提高效率
                for index, line in enumerate(csv_reader):
                    line.insert(0, str(index+1))
                    rows_list.append(line)
                    count_data = index+1
                data_export = pd.DataFrame(rows_list, columns=['序号', '工号', '姓名', '打卡时间'])
            data_export.to_excel(filepath, index=False)
            QMessageBox.information(None, "完成", '共计'+str(count_data)+'条数据。\n已成功导出至：'+str(filepath),
                                    QMessageBox.Yes, QMessageBox.Yes)
        except Exception as e:
            QMessageBox.critical(None, "失败", '发生错误，导出失败:'+str(e),
                                    QMessageBox.Yes, QMessageBox.Yes)


def back_menu():
    # 返回主界面
    global cap, ui
    timer_attendance_show_pic.stop()
    timer_register_show_pic.stop()
    cap.release()
    ui = Ui_menuWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.button_register.clicked.connect(open_register)
    ui.button_attendance.clicked.connect(open_attendance)
    ui.button_management.clicked.connect(open_management)


def open_register():
    # 打开注册界面
    global ui
    ui = Ui_registerWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.button_register_start.clicked.connect(register_start)
    ui.button_register_cancel.clicked.connect(back_menu)


def open_attendance():
    # 打开考勤界面
    global ui, cap, feature_list, label_list, name_list
    feature_list, label_list, name_list = get_data_list()
    ui = Ui_attendanceWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.button_attendance_start.clicked.connect(attendance_start)
    ui.button_attendance_back.clicked.connect(back_menu)
    cap = cv2.VideoCapture(0)
    timer_attendance_show_pic.start(5)


def open_management():
    # 打开信息管理界面
    global ui
    ui = Ui_managementWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.button_export.clicked.connect(export_infor)
    ui.button_back.clicked.connect(back_menu)
    ui.table_data.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    with open('./data/log_attendance.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for index, line in enumerate(csv_reader):
            # 重新加载数据
            row_count = ui.table_data.rowCount()  # 返回当前行数(尾部)
            ui.table_data.insertRow(row_count) # 在尾部插入一行
            ui.table_data.setItem(index, 0, QTableWidgetItem(line[0]))
            ui.table_data.setItem(index, 1, QTableWidgetItem(line[1]))
            ui.table_data.setItem(index, 2, QTableWidgetItem(line[2]))


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
    timer_attendance_show_pic.timeout.connect(show_camera)
    sys.exit(app.exec_())
