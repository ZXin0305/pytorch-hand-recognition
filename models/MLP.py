import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd
import numpy
from torch.utils.data import DataLoader,Dataset,random_split,Subset
from torchsummary import summary
from IPython import embed
import os


class MLP(nn.Module):
    def __init__(self, in_ch=21 * 2, num_cls=16):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_ch, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, num_cls)
        self.ac = nn.ReLU()

    def forward(self,x):
        out = self.fc1(x)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.ac(out)
        out = self.fc3(out)
        return out
    
if __name__ == "__main__":
    input = torch.randn(size=(1,42))
    model = MLP(42, 11)
    pre = model(input)