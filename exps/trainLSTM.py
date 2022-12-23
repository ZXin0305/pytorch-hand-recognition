import sys
# from typing_extensions import Required
sys.path.append("/home/xuchengjun/ZXin/pytorch-hand-recognition")
import argparse
import time
import torch.nn as nn
import torch
import numpy as np
from tensorboardX import SummaryWriter
from lib.tools import *
from models.LSTM import LSTM_model
from lib.dataloader import get_train_loader, get_test_loader
from path import Path
from lib.solver import make_lr_scheduler, make_optimizer
from IPython import embed
import random
import os
from config.config import cfg
from lib.logger import *
from torch.utils.data import DataLoader
from dataset.dataset import DatasetV2


def train(model, current_epoch, optimizer, scheduler, iter, train_loader, loss_fn):

    train_loss = []
    max_iter = len(train_loader)

    time1 = time.time()
    avg_loss = None
    for img, label in train_loader:
        optimizer.zero_grad()
        img = img.to(cfg.MODEL.DEVICE)
        label = label.to(cfg.MODEL.DEVICE)
        
        output = model(img)
        total_loss = loss_fn(output, label.long())

        total_loss.backward(retain_graph=True)
        optimizer.step()

        train_loss.append(total_loss.data.item())

        if iter % 20 == 0 or iter == max_iter:
            avg_loss = np.mean(np.array(train_loss))
            print('Epoch: {} process: {}/{} Loss: {} LR: {}'.format(current_epoch, iter, 
                                                                    len(train_loader), 
                                                                    avg_loss,
                                                                    optimizer.param_groups[0]["lr"])
                                                                    )
        iter += 1
        if cfg.RUN_EFFICIENT:
            del img, label, total_loss

        # # save ck
        # if iter % cfg.MODEL.CHECK_PERIOD == 0:
        #     ck = {
        #         'state_dict': model.module.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'iter': iter,
        #         'scheduler': scheduler.state_dict(),
        #         'epoch': current_epoch
        #     }
        #     torch.save(ck, Path(cfg.MODEL.CHECKPOINT_PATH) / f'checkpoint.pth')
        # # save weights
        # if current_epoch != 0 and current_epoch % cfg.MODEL.SAVE_PERIOD == 0:
        #     torch.save(model.module.state_dict(), Path(cfg.MODEL.SAVE_PATH) / f'MLP_{current_epoch}.pth')
    
    scheduler.step()
    if len(train_loss) != 0:
        return avg_loss

def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(cfg.MODEL.DEVICE) 

            output = model(img)
            pre = output.argmax(1).detach().cpu()

            if pre == label:
                correct += 1
        print('this epoch acc is: %0.5f' % (correct / len(test_loader)))
    return correct / len(test_loader)

def main():
    # ---------------- #
    # init params
    start_epoch = 1
    end_epoch = 23
    best_acc = 0
    best_epoch = 0
    iter = 1
    loss_fn = nn.CrossEntropyLoss()
    ensure_dir(cfg.MODEL.SAVE_PATH)  # 创建文件夹

    model = LSTM_model(42,42,6,4, 0,11,1,42)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model, num_gpu=cfg.MODEL.GPU_NUM)
    scheduler = make_lr_scheduler(cfg, optimizer)

    assert optimizer != None
    assert scheduler != None
    
    train_dataset = DatasetV2(cfg, 'train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=2, drop_last=True)
    test_dataset = DatasetV2(cfg, 'test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=0, drop_last=False)

    # load ck
    if Path(cfg.MODEL.WEIGHT).exists():
        print(f'load checkpoint --> {cfg.MODEL.WEIGHT} ..')
        ck = torch.load(cfg.MODEL.WEIGHT)

        model.load_state_dict(ck["state_dict"])
        print('load model state !')

        optimizer.load_state_dict(ck["optimizer"])
        print('loaded optimizer state !')

        iter = ck["iter"]

        scheduler.load_state_dict(ck["scheduler"])
        print('loaded scheduler state !')

        if iter == len(train_loader):
            start_epoch = ck['epoch'] + 1
            iter = 0
        else:
            start_epoch = ck['epoch']
        
    # ------------------------------- 

    if cfg.MODEL.DP:
        print('using dp model ..')
        model = torch.nn.DataParallel(model, device_ids=cfg.MODEL.GPU_IDS)

    print('training process beginning ..')
    model.train()
    for current_epoch in range(start_epoch, end_epoch):
        avg_loss = train(model, current_epoch, optimizer, scheduler, iter, train_loader, loss_fn)
        print("==================================================")
        print('testing ..')
        acc = test(model, test_loader)
        print("==================================================")
        if acc >= best_acc:
            best_acc = acc
            best_epoch = current_epoch
            torch.save(model.module.state_dict(), Path(cfg.MODEL.BEST_PATH))

        iter = 1       # reset iter ..
        model.train()  # open train mode ..
    print('training process finish !!!')
    print(f"best acc: {best_acc}, best epoch: {best_epoch}")

if __name__ == "__main__":
    main()