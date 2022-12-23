"""
这个是用来训练SVM的
用skel的模态
"""
import sys
sys.path.append("/home/xuchengjun/ZXin/pytorch-hand-recognition")
from symbol import test_nocond
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import pickle
from lib.tools import *
from os import path as osp
from IPython import embed

# svm
from sklearn import svm
# knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, minmax_scale
#decisionTree
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier,export_graphviz
#贝叶斯
from sklearn.naive_bayes import GaussianNB
#Linear
from sklearn.linear_model import LinearRegression
#LDA  线性判别分析
from sklearn.decomposition import LatentDirichletAllocation
#kernel
from sklearn.kernel_ridge import KernelRidge


class TrainSVM():
    def __init__(self, dataset_root, train_json_file, test_json_file) -> None:
        self.dataset_root = dataset_root
        self.train_file = train_json_file
        self.test_file = test_json_file

    def getData(self):
        train_feature_list = []
        train_label_list = []

        test_feature_list = []
        test_label_list = []
        
        train_data_list = read_json(self.train_file)['data']
        test_data_list = read_json(self.test_file)['data']
        print('begin process dataset ..')
        for i in range(len(train_data_list)):
            data_info = train_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            train_feature_list.append(input)
            train_label_list.append(gesture_label)
            print(f'working train list --> {i + 1}/{len(train_data_list)}')

        for i in range(len(test_data_list)):
            data_info = test_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            test_feature_list.append(input)
            test_label_list.append(gesture_label)
            print(f'working test list --> {i + 1}/{len(test_data_list)}')
            
        return np.array(train_feature_list), np.array(test_feature_list), np.array(train_label_list), np.array(test_label_list)
    
    def train(self):
        X_train, X_test, Y_train, Y_test = self.getData()  # train_data, test_data, train_label, test_label
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        clf = svm.LinearSVC(C=11, max_iter=200000)
        
        print('begin training process ..')
        clf.fit(X_train, Y_train)  # feature,  label ..
        print('ended training ..')
        pred = clf.predict(X_test)
    
        correct = 0
        for i in range(len(X_test)):
            if int(pred[i]) == int(Y_test[i]):
                correct += 1
        print(f'acc: {correct / len(pred)}')

class KNN():
    def __init__(self, dataset_root, train_json_file, test_json_file) -> None:
        self.dataset_root = dataset_root
        self.train_file = train_json_file
        self.test_file = test_json_file

    def getData(self):
        train_feature_list = []
        train_label_list = []

        test_feature_list = []
        test_label_list = []
        
        train_data_list = read_json(self.train_file)['data']
        test_data_list = read_json(self.test_file)['data']
        print('begin process dataset ..')
        for i in range(len(train_data_list)):
            data_info = train_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            train_feature_list.append(input)
            train_label_list.append(gesture_label)
            print(f'working train list --> {i + 1}/{len(train_data_list)}')

        for i in range(len(test_data_list)):
            data_info = test_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            test_feature_list.append(input)
            test_label_list.append(gesture_label)
            print(f'working test list --> {i + 1}/{len(test_data_list)}')
            
        return np.array(train_feature_list), np.array(test_feature_list), np.array(train_label_list), np.array(test_label_list)
    
    def train(self):
        X_train, X_test, Y_train, Y_test = self.getData()  # train_data, test_data, train_label, test_label
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        print('begin training process ..')
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, Y_train)
        print('ended training ..')
        
        y_predit = knn.predict(X_test)
        
        correct = 0
        for i in range(len(X_test)):
            if int(y_predit[i]) == int(Y_test[i]):
                correct += 1
        print(f'acc: {correct / len(y_predit)}')

class DecisionTree():
    def __init__(self, dataset_root, train_json_file, test_json_file) -> None:
        self.dataset_root = dataset_root
        self.train_file = train_json_file
        self.test_file = test_json_file

    def getData(self):
        train_feature_list = []
        train_label_list = []

        test_feature_list = []
        test_label_list = []
        
        train_data_list = read_json(self.train_file)['data']
        test_data_list = read_json(self.test_file)['data']
        print('begin process dataset ..')
        for i in range(len(train_data_list)):
            data_info = train_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            train_feature_list.append(input)
            train_label_list.append(gesture_label)
            print(f'working train list --> {i + 1}/{len(train_data_list)}')

        for i in range(len(test_data_list)):
            data_info = test_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            test_feature_list.append(input)
            test_label_list.append(gesture_label)
            print(f'working test list --> {i + 1}/{len(test_data_list)}')
            
        return np.array(train_feature_list), np.array(test_feature_list), np.array(train_label_list), np.array(test_label_list)
    
    def train(self):
        X_train, X_test, Y_train, Y_test = self.getData()  # train_data, test_data, train_label, test_label
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        
        print('begin training process ..')
        dc = DecisionTreeClassifier()
        dc.fit(X_train, Y_train)
        print('ended training ..')
        
        print(dc.score(X_test, Y_test))

class NaiveBayes():
    def __init__(self, dataset_root, train_json_file, test_json_file) -> None:
        self.dataset_root = dataset_root
        self.train_file = train_json_file
        self.test_file = test_json_file

    def getData(self):
        train_feature_list = []
        train_label_list = []

        test_feature_list = []
        test_label_list = []
        
        train_data_list = read_json(self.train_file)['data']
        test_data_list = read_json(self.test_file)['data']
        print('begin process dataset ..')
        for i in range(len(train_data_list)):
            data_info = train_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            train_feature_list.append(input)
            train_label_list.append(gesture_label)
            print(f'working train list --> {i + 1}/{len(train_data_list)}')

        for i in range(len(test_data_list)):
            data_info = test_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            test_feature_list.append(input)
            test_label_list.append(gesture_label)
            print(f'working test list --> {i + 1}/{len(test_data_list)}')
            
        return np.array(train_feature_list), np.array(test_feature_list), np.array(train_label_list), np.array(test_label_list)
    
    def train(self):
        X_train, X_test, Y_train, Y_test = self.getData()  # train_data, test_data, train_label, test_label
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        
        print('begin training process ..')
        dc = GaussianNB()
        dc.fit(X_train, Y_train)
        print('ended training ..')
        
        print(dc.score(X_test, Y_test))

class Linear():
    def __init__(self, dataset_root, train_json_file, test_json_file) -> None:
        self.dataset_root = dataset_root
        self.train_file = train_json_file
        self.test_file = test_json_file

    def getData(self):
        train_feature_list = []
        train_label_list = []

        test_feature_list = []
        test_label_list = []
        
        train_data_list = read_json(self.train_file)['data']
        test_data_list = read_json(self.test_file)['data']
        print('begin process dataset ..')
        for i in range(len(train_data_list)):
            data_info = train_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            train_feature_list.append(input)
            train_label_list.append(gesture_label)
            print(f'working train list --> {i + 1}/{len(train_data_list)}')

        for i in range(len(test_data_list)):
            data_info = test_data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            if is_aug_data:
                continue
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            test_feature_list.append(input)
            test_label_list.append(gesture_label)
            print(f'working test list --> {i + 1}/{len(test_data_list)}')
            
        return np.array(train_feature_list), np.array(test_feature_list), np.array(train_label_list), np.array(test_label_list)
    
    def train(self):
        X_train, X_test, Y_train, Y_test = self.getData()  # train_data, test_data, train_label, test_label
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        
        print('begin training process ..')
        dc = LinearRegression()
        dc.fit(X_train, Y_train)
        print('ended training ..')
        
        print(dc.score(X_test, Y_test))

class LDA():
    def __init__(self, dataset_root, json_file) -> None:
        self.dataset_root = dataset_root
        self.file = json_file
        pass
    
    def getData(self):
        feature_list = []
        label_list = []
        
        data_list = read_json(self.file)['data']
        print('begin process dataset ..')
        for i in range(len(data_list)-30000):
            data_info = data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            feature_list.append(input)
            label_list.append(gesture_label)
            print(f'working {i + 1}/{len(data_list)}')
            
        return np.array(feature_list), np.array(label_list)

    def train(self):
        feature_list, label_list = self.getData()
        X_train, X_test, Y_train, Y_test = train_test_split(feature_list, label_list, test_size=0.2)
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        
        print('begin training process ..')
        dc = LatentDirichletAllocation(n_components=11, random_state=0)
        dc.fit_transform(X_train, Y_train)
        print('ended training ..')
        
        print(dc.score(X_test, Y_test))
        
class Kernel():
    def __init__(self, dataset_root, json_file) -> None:
        self.dataset_root = dataset_root
        self.file = json_file
        pass
    
    def getData(self):
        feature_list = []
        label_list = []
        
        data_list = read_json(self.file)['data']
        print('begin process dataset ..')
        for i in range(len(data_list)):
            data_info = data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            
            gesture_label = int(data_info[3])
            skel_ori = read_csv(osp.join(self.dataset_root, skel_file))
            
            skel_numpy = np.array(skel_ori, dtype=np.float)[0:21, 1:]
            input = skel_numpy
            input = input.flatten()
            
            feature_list.append(input)
            label_list.append(gesture_label)
            print(f'working {i + 1}/{len(data_list)}')
            
        return np.array(feature_list), np.array(label_list)

    def train(self):
        feature_list, label_list = self.getData()
        X_train, X_test, Y_train, Y_test = train_test_split(feature_list, label_list, test_size=0.2)
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        
        print('begin training process ..')
        dc = KernelRidge(alpha=1.0)
        dc.fit(X_train, Y_train)
        print('ended training ..')
        
        print(dc.score(X_test, Y_test))
        
    
        
if __name__ == "__main__":
    # /media/xuchengjun/disk1/zx/left/other_rate     /media/xuchengjun/disk1/zx/left/other_rate/left_train(1:9).json
    # /media/xuchengjun/disk1/zx/left/other_rate/left_test(1:9).json
    # /media/xuchengjun/disk1/zx/left/left_train.json    /media/xuchengjun/disk1/zx/left/left_test.json
    # (1:9)
    dataset_root = "/media/xuchengjun/disk1/zx/right"
    train_json_file = "/media/xuchengjun/disk1/zx/right/other_rate/right_test(1:9).json"
    test_json_file = "/media/xuchengjun/disk1/zx/right/other_rate/right_train(1:9).json"
    # trainer = TrainSVM(dataset_root=dataset_root, train_json_file=train_json_file, test_json_file=test_json_file)
    # trainer.train()

    # trainer = KNN(dataset_root=dataset_root, train_json_file=train_json_file, test_json_file=test_json_file)
    # trainer.train()

    # trainer = DecisionTree(dataset_root=dataset_root, train_json_file=train_json_file, test_json_file=test_json_file)
    # trainer.train()

    # trainer = NaiveBayes(dataset_root=dataset_root, train_json_file=train_json_file, test_json_file=test_json_file)
    # trainer.train()

    trainer = Linear(dataset_root=dataset_root, train_json_file=train_json_file, test_json_file=test_json_file)
    trainer.train()
    
    # trainer = LDA(dataset_root=dataset_root, json_file=json_file)
    # trainer.train()