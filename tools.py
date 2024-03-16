import numpy as np
import csv


# 返回DLIB格式的face
def get_dlib_rect(face=None):
    l, t, r, b = None, None, None, None
    l = face[0]
    t = face[1]
    r = face[0] + face[2]
    b = face[1] + face[3]
    nonnegative = lambda x: x if x >= 0 else 0
    return map(nonnegative, (l, t, r, b))


# 获取CSV中信息
def get_data_list():
    feature_list = None
    label_list = []
    name_list = []
    # 加载保存的特征样本
    try:
        with open('./data/face_feature.csv', 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                # 重新加载数据
                faceId = line[0]
                userName = line[1]
                face_descriptor = eval(line[2])
                label_list.append(faceId)
                name_list.append(userName)
                # 转为numpy格式
                face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
                # 转为二维矩阵，拼接
                face_descriptor = np.reshape(face_descriptor, (1, -1))
                # 初始化
                if feature_list is None:
                    feature_list = face_descriptor
                else:
                    # 拼接
                    feature_list = np.concatenate((feature_list, face_descriptor), axis=0)
    except Exception as e:
        print('特征加载失败:'+str(e))
    return feature_list, label_list, name_list
