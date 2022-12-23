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



# class TrainSVM():
#     def __init__(self) -> None:
#         self.dataset_root = "/media/xuchengjun/disk1/zx"
#         self.input_shape = (160, 160)
#         self.file = "/media/xuchengjun/disk1/zx/HAND/new_train.json"
#         self.epoches = 2  # 没有用

#     def getImageData(self):
#         feature_list = []
#         label_list = []
        
#         data_list = read_json(self.file)['data']
#         print('begin process dataset ..')
#         for i in range(len(data_list)):
#             data_info = data_list[i]
#             rgb_file = data_info[0]
#             depth_file = data_info[1]
#             emg_file = data_info[2]
#             gesture_label = int(data_info[3])
            
#             rgb_ori = cv2.imread(osp.join(self.dataset_root, rgb_file), cv2.IMREAD_COLOR)
#             depth_ori = cv2.imread(osp.join(self.dataset_root, depth_file))
#             emg_ori= read_csv(osp.join(self.dataset_root, emg_file))
            
#             emg_numpy = np.array(emg_ori, dtype=np.float)
            
#             if emg_numpy.shape[0] <= 1:   
#                 emg_numpy = np.ones(shape=(8, 8), dtype=np.float) * 0.5  
                
#             emg_map = emg_mapping(emg_numpy)
#             emg = cv2.resize(emg_map, (self.input_shape[0], self.input_shape[1]))
#             emg_mean = (emg / np.mean(emg))[np.newaxis, :, :]
            
#             depth = depth_mapping(depth_ori)[:, :, 0]
#             depth_mean_val = np.mean(depth) if np.mean(depth) > 0 else 1
#             depth_mean = (depth / depth_mean_val)[np.newaxis, :, :]
            
#             rgb_mean = (rgb_ori / np.mean(rgb_ori)).transpose((2, 1, 0))
            
#             input = np.concatenate((rgb_mean, depth_mean, emg_mean), 0)
#             input = input.flatten()
            
#             feature_list.append(input)
#             label_list.append(gesture_label)
#             print(f'working {i + 1}/{len(data_list)}')
            
#         return np.array(feature_list), np.array(label_list)
    
    
#     def train(self):
#         feature_list, label_list = self.getImageData()
#         X_train, X_test, Y_train, Y_test =  train_test_split(feature_list, label_list, test_size=0.3, random_state=42)
#         print('have processed dataset ..')
#         clf = svm.SVC(gamma=0.0001, C=17, probability=True)
        
#         print('begin training process ..')
#         clf.fit(X_train, Y_train)  # feature,  label ..
#         print('ended training ..')
#         pred = clf.predict(X_test)
    
#         correct = 0
#         for i in range(len(X_test)):
#             if int(pred[i]) == int(Y_test[i]):
#                 correct += 1
        
#         print(f'acc: {correct / len(pred)}')


class TrainSVM():
    def __init__(self, dataset_root, json_file) -> None:
        self.dataset_root = dataset_root
        self.file = json_file

    def getData(self):
        feature_list = []
        label_list = []
        
        data_list = read_json(self.file)['data']
        print('begin process dataset ..')
        for i in range(len(data_list)):
            data_info = data_list[i]
            img_file = data_info[0]
            skel_file = data_info[2]
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            # if is_aug_data:
            #     continue
            
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
        X_train, X_test, Y_train, Y_test = train_test_split(feature_list, label_list, test_size=0.2, random_state=35)
        embed()
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        clf = svm.LinearSVC(C=11)
        
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
            is_aug_data = img_file.split("/")[3] == 'aug_data'
            
            # if is_aug_data:
            #     continue
            
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
        X_train, X_test, Y_train, Y_test = train_test_split(feature_list, label_list, test_size=0.3)
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        
        print('begin training process ..')
        dc = DecisionTreeClassifier()
        dc.fit(X_train, Y_train)
        print('ended training ..')
        
        print(dc.score(X_test, Y_test))

class NaiveBayes():
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
        dc = GaussianNB()
        dc.fit(X_train, Y_train)
        print('ended training ..')
        
        print(dc.score(X_test, Y_test))

class Linear():
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
    dataset_root = "/media/xuchengjun/disk1/zx/left"
    json_file = "/media/xuchengjun/disk1/zx/left/final.json"
    trainer = TrainSVM(dataset_root=dataset_root, json_file=json_file)
    trainer.train()

    # trainer = KNN(dataset_root=dataset_root, json_file=json_file)
    # trainer.train()

    # trainer = DecisionTree(dataset_root=dataset_root, json_file=json_file)
    # trainer.train()

    # trainer = NaiveBayes(dataset_root=dataset_root, json_file=json_file)
    # trainer.train()

    # trainer = Linear(dataset_root=dataset_root, json_file=json_file)
    # trainer.train()
    
    # trainer = LDA(dataset_root=dataset_root, json_file=json_file)
    # trainer.train()
    
    
           
            
            
            
            
            
            
            
        