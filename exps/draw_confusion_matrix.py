import torch
import sys
sys.path.append("/home/xuchengjun/ZXin/pytorch-hand-recognition")
import numpy as np
from lib.tools import *
from lib.get_net import get_network
from config.config import cfg
from lib.dataloader import get_train_loader, get_test_loader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import time

# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

#     plt.figure(figsize=(12, 8), dpi=100)
#     np.set_printoptions(precision=3)

#     # 在混淆矩阵中每格的概率值
#     ind_array = np.arange(len(classes))
#     x, y = np.meshgrid(ind_array, ind_array)
#     for x_val, y_val in zip(x.flatten(), y.flatten()):
#         c = cm[y_val][x_val]
#         if c >= 0.00001:
#             plt.text(x_val, y_val, "%0.3f" % (c,), color='white', fontsize=8, va='center', ha='center')
#         if c == 0:
#             plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=8, va='center', ha='center')

#         # plt.text(x_val, y_val, "%d" % (c,), color='white', fontsize=8, va='center', ha='center')


#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(title)
#     plt.colorbar()
#     xlocations = np.array(range(len(classes)))
#     plt.xticks(xlocations, classes, rotation=90)
#     plt.yticks(xlocations, classes, rotation=30)
#     plt.ylabel('Actual')
#     plt.xlabel('Predict')
    
#     # offset the tick
#     tick_marks = np.array(range(len(classes)))
#     plt.gca().set_xticks(tick_marks, minor=True)
#     plt.gca().set_yticks(tick_marks, minor=True)
#     plt.gca().xaxis.set_ticks_position('none')
#     plt.gca().yaxis.set_ticks_position('none')
#     plt.grid()  # True, which='minor', linestyle='-'
#     plt.gcf().subplots_adjust(bottom=0.15)
    
#     # show confusion matrix
#     plt.savefig(savename, format='png')
#     plt.show()

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        # if c > 0.001:
        #     plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
        # if c == 0:
        #     plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
        # if c > 0.5:
        #     plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=8, va='center', ha='center')
        # elif c> 0.01:
        #     plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=8, va='center', ha='center')
        if c == 0:
            plt.text(x_val, y_val, "%0.1f" % (c,), color='black', fontsize=8, va='center', ha='center')
        elif c < 0.1:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=8, va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='white', fontsize=8, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes, rotation=30)
    plt.ylabel('Actual action')
    plt.xlabel('Predict action')
    
    # offset the tick
    tick_marks = np.array(range(len(classes)))
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

if __name__ == "__main__":
    device = 'cuda:0'
    weight_path = '/media/xuchengjun/disk1/zx/model/right/right_multi(B=3).pth'
    # dataset_path = '/media/xuchengjun/disk1/zx/left/left_test.json'
    # /media/xuchengjun/disk1/zx/model/right
    model = get_network(cfg)
    device = torch.device(device)
    model.to(device)
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    model.eval()

    test_loader = get_test_loader(cfg)

    i = 0
    total = len(test_loader)
    correct = 0
    out_pre = []
    label_ = []
    time_list = []
    len_data = len(test_loader)

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            st = time.time()
            output = model(data)
            time_list.append(time.time() - st)
            pre_label = int(output.argmax(1).detach().cpu())

            if pre_label == label:
                correct += 1

            out_pre.append(pre_label)
            label_.append(int(label))

            i += 1
            # if (i >= 100):
            #     break
            # print(f"pre .. {len(test_loader)} / {i}")
    
        # out_pre = np.array(out_pre)
        # label_ = np.array(label_)

        # calculate Precision \ Recall \ F1-Score
        f1 = f1_score(label_, out_pre, average='macro')
        pre = precision_score(label_, out_pre, average='macro')
        re = recall_score(label_, out_pre, average='macro')
        print(f'precision: {pre}  recall: {re}  f1-score: {f1}')

        # draw confusion matrix
        # cm = confusion_matrix(label_, out_pre)
        # cm_normalized = cm.astype('double') / cm.sum(axis=1)[:, np.newaxis]
        # plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')
        
        print(f'acc: {correct / len(test_loader)}')

        avg_time = np.sum(np.array(time_list, np.float)) / len_data
        print(avg_time)

            

    