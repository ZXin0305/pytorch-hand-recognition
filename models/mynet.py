"""_summary_
Myself net for hand gesture classify 
"""

from turtle import forward
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from IPython import embed
from inceptionv3 import *

class conv_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
                 has_bn=True, has_relu=True, efficient=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.efficient = efficient
        
    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func
        
        func = _func_factory(self.conv, self.bn, self.relu, self.has_bn, self.has_relu)
        
        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)
        return x
    
class PryBottleNet(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, scaled):
        super().__init__()
        self.ori_shape = ori_shape
        self.max_pool = nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(scaled, scaled))
        self.conv1 = conv_bn_relu(out_ch, out_ch, kernel_size=3, stride=1, padding=1, has_bn=True, has_relu=True)
        self.conv2 = conv_bn_relu(out_ch, out_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=True)
        
    def forward(self, x):
        out = self.max_pool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = F.interpolate(out, size=self.ori_shape, mode='bilinear', align_corners=True)
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, type, cardinality):
        super(ConvBlock, self).__init__()
        self.middle_ch = out_ch // 2
        self.C = cardinality  # the num of prm branch, default is 3
        self.ori_shape = ori_shape
        self.pyramid = list()
        # self.scaled = 2 ** (1 / self.C)  # control the scaled ratio
        self.scaled = 2
        self.type = type

        self.main_branch = nn.Sequential()
        if self.type != 'no_preact':
            self.main_branch.add_module('activation_layer1', nn.BatchNorm2d(in_ch))
            self.main_branch.add_module('activation_layer2', nn.ReLU(inplace=True))

        self.main_branch.add_module('conv1', conv_bn_relu(in_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=True))
        self.main_branch.add_module('conv2', conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=3, stride=1, padding=1, has_relu=True, has_bn=True))
        self.main_branch.add_module('conv3', conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False))

        for i in range(self.C):
            tmp_scaled = 1 / (self.scaled ** (i+1))  # change the scaled to change the feature resolution
            self.pyramid.append(
                PryBottleNet(self.middle_ch, self.middle_ch, self.ori_shape, tmp_scaled)
            )

            setattr(self, 'pry{}'.format(i), self.pyramid[i])

        self.conv_top = conv_bn_relu(in_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)  # 这个是在pra各个等级之前的，进行一次卷积
        self.bn = nn.BatchNorm2d(self.middle_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv_bot = conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)  #这个是紧跟着pra输出相加之后的卷积
        self.conv_out = conv_bn_relu(self.middle_ch, out_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)   #这个是再一次进行卷积

    def forward(self, x):
        # 1. main branch
        out_main = self.main_branch(x)  # ori shape in, ori shape out

        # 2. pyramid branch
        # ---------------------------
        out_pry = None
        pyraTable = list()
        conv_top = self.conv_top(x)
        for i in range(self.C):
            out_pry = eval('self.pry' + str(i))(conv_top)   # 这里出来的都是和输入尺寸一样的
            pyraTable.append(out_pry)
            if i != 0:
                out_pry = pyraTable[i] + pyraTable[0]

        out_pry = self.bn(out_pry)        # 在前面使用bn和relu是为了减少方差
        out_pry = self.relu(out_pry)

        out_pry = self.conv_bot(out_pry)  # 金字塔分支进行相加后卷积

        # ------------------------------
        assert out_pry.shape == out_main.shape
        out = out_pry + out_main
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv_out(out)

        return out
    
class SkipLayer(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(SkipLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv_bn_relu(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, has_bn=False, has_relu=False)

    def forward(self, x):
        if self.in_ch == self.out_ch:
            out = x
        else:
            out = self.bn(x)
            out = self.relu(out)
            out = self.conv(out)
        return out
    
class PRM(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, cnf, type):
        """
        :param in_ch: 
        :param out_ch: 
        :param cnf:
        """
        cardinality = 2   #这个数？？？？
        super(PRM, self).__init__()
        self.skip_layer = SkipLayer(in_ch, out_ch, stride=1)                      # 主要的原分支
        self.pry_layer = ConvBlock(in_ch, out_ch, ori_shape, type, cardinality)   #　多分辨率分支
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_skip = self.skip_layer(x) 
        out_pry = self.pry_layer(x)
        assert out_pry.shape == out_skip.shape
        out = out_pry + out_skip
        out = self.bn(out)
        out = self.relu(out)  
        return out 


class MyNet(nn.Module):
    def __init__(self, in_ch, out_ch, feature_shape=(160, 160), num_classes=17):
        super().__init__()
        # 1.RGB + depth
        self.up_net_head = PRM(in_ch, out_ch, feature_shape, None, 'no_preact')
        self.max_pool_up = nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(3 / 16, 3 / 16))
        
        #6*6 feature size
        self.avgpool_up = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_up = nn.Dropout2d()
        self.linear_up = nn.Linear(2048, num_classes)
        

        # bottom  --> ori inceptionv3
        self.avg_pool_bot = nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(1 / 5, 1 / 5))
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        #naive inception module
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        #downsample
        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        #downsample
        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        #6*6 feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(2048, num_classes)
        
    def forward(self, x, y):
        # up
        x_out = self.up_net_head(x)
        x_out = self.max_pool_up(x_out)
        x_out = self.Mixed_5b(x_out)
        x_out = self.Mixed_5c(x_out)
        x_out = self.Mixed_5d(x_out)

        #downsample
        x_out = self.Mixed_6a(x_out)

        x_out = self.Mixed_6b(x_out)
        x_out = self.Mixed_6c(x_out)
        x_out = self.Mixed_6d(x_out)
        x_out = self.Mixed_6e(x_out)

        #downsample
        x_out = self.Mixed_7a(x_out)

        x_out = self.Mixed_7b(x_out)
        x_out = self.Mixed_7c(x_out)

        #6*6 feature size
        x_out = self.avgpool_up(x_out)
        x_out = self.dropout_up(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.linear_up(x_out)      
        
        # bot
        # 32 -> 30
        y_out = self.avg_pool_bot(y)
        y_out = self.Conv2d_1a_3x3(y_out)
        y_out = self.Conv2d_2a_3x3(y_out)
        y_out = self.Conv2d_2b_3x3(y_out)
        y_out = self.Conv2d_3b_1x1(y_out)
        y_out = self.Conv2d_4a_3x3(y_out)

        #30 -> 30
        y_out = self.Mixed_5b(y_out)
        y_out = self.Mixed_5c(y_out)
        y_out = self.Mixed_5d(y_out)

        #30 -> 14
        y_out = self.Mixed_6a(y_out)

        #14 -> 14
        y_out = self.Mixed_6b(y_out)
        y_out = self.Mixed_6c(y_out)
        y_out = self.Mixed_6d(y_out)
        y_out = self.Mixed_6e(y_out)

        #14 -> 6
        y_out = self.Mixed_7a(y_out)

        #6 -> 6
        y_out = self.Mixed_7b(y_out)
        y_out = self.Mixed_7c(y_out)

        #6 -> 1
        y_out = self.avgpool(y_out)
        y_out = self.dropout(y_out)
        y_out = y_out.view(y_out.size(0), -1)
        y_out = self.linear(y_out)        
        
        return x_out, y_out
    
    
if __name__ == "__main__":
    # net = PRM(4, 64, (160, 160), None, 'no_preact')
    net = MyNet(4, 192, (160, 160)).to("cuda:0")
    from time import time
    input1 = torch.ones(size=(1, 4, 160 ,160), dtype=torch.float32).to("cuda:0")
    input2 = torch.ones(size=(1, 1, 160 ,160), dtype=torch.float32).to("cuda:0")
    st = time()
    x_out, y_out = net(input1, input2)
    et = time()
    print(et - st)
    # embed()
