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
from lib.get_net import get_network
from lib.dataloader import get_train_loader, get_test_loader
from path import Path
from lib.solver import make_lr_scheduler, make_optimizer
from IPython import embed
import random
import os
from config.config import cfg
from lib.logger import *

logger = None

def train(model, current_epoch, optimizer, scheduler, iter, train_loader, loss_fn):
    global logger
    logger = setup_log(name='train', log_dir=cfg.LOG_DIR, file_name=cfg.LOG_FILE_NAME)
    ensure_dir(cfg.LOG_DIR)
    train_loss = []
    max_iter = len(train_loader)

    time1 = time.time()
    avg_loss = None
    for img, label in train_loader:
        optimizer.zero_grad()
        img = img.to(cfg.MODEL.DEVICE)
        
        # depth = depth.to(cfg.MODEL.DEVICE)
        # emg = emg.to(cfg.MODEL.DEVICE)

        label = label.to(cfg.MODEL.DEVICE)
        
        output = model(img)
        # multimodal
        # loss1 = loss_fn(output[0], label.long())
        # loss2 = loss_fn(output[1], label.long())
        # total_loss = 0.6 * loss1 + 0.4 * loss2 

        # single modal
        total_loss = loss_fn(output, label.long())

        total_loss.backward(retain_graph=True)
        optimizer.step()

        train_loss.append(total_loss.data.item())

        if iter % 20 == 0 or iter == max_iter:
            avg_loss = np.mean(np.array(train_loss))
            # print('Epoch: {} process: {}/{} Loss: {} LR: {}'.format(current_epoch, iter, 
            #                                                         len(train_loader), 
            #                                                         avg_loss,
            #                                                         optimizer.param_groups[0]["lr"]),
            #                                                         end='')
            log_str = 'Epoch:%d, Iter:%d, Max_iter:%d, LR:%.1e, Loss:%0.5e, ' % (
                current_epoch ,iter, max_iter, optimizer.param_groups[0]["lr"], avg_loss)

            time2 = time.time()
            elapased_time = time2 - time1
            time1 = time2
            required_time = elapased_time / 20 * (max_iter - iter)
            hours = required_time // 3600
            mins = required_time % 3600 // 60
            # print(f'need finish --> {hours} hours and {mins} mins ..')
            log_str += 'This epoch finish: %dh%dmin, ' % (hours, mins)
            logger.info(log_str)

        iter += 1
        if cfg.RUN_EFFICIENT:
            del img, label, total_loss

        # save ck
        if iter % cfg.MODEL.CHECK_PERIOD == 0:
            ck = {
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter,
                'scheduler': scheduler.state_dict(),
                'epoch': current_epoch
            }
            torch.save(ck, Path(cfg.MODEL.CHECKPOINT_PATH) / f'checkpoint.pth')
        # save weights
        if current_epoch != 0 and current_epoch % cfg.MODEL.SAVE_PERIOD == 0:
            torch.save(model.module.state_dict(), Path(cfg.MODEL.SAVE_PATH) / f'{cfg.MODEL.NAME}_{current_epoch}.pth')
        
        if iter >= len(train_loader):
            break
          
    scheduler.step()
    if len(train_loss) != 0:
        return avg_loss

def test(model, test_loader):
    global logger
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(cfg.MODEL.DEVICE) 
            # emg = emg.to(cfg.MODEL.DEVICE)

            output = model(img)
            # output = (output[0] + output[1]) / 2.0  # multimodal
            pre = output.argmax(1).detach().cpu()

            if pre == label:
                correct += 1
        log_str = 'this epoch acc is: %0.5f' % (correct / len(test_loader))
        logger.info(log_str)
    return correct / len(test_loader)

def main():
    # ---------------- #
    # init params
    start_epoch = cfg.START_EPOCH
    end_epoch = cfg.END_EPOCH
    best_acc = 0
    best_epoch = 0
    iter = 1
    loss_fn = nn.CrossEntropyLoss()
    tb_writer = SummaryWriter(cfg.TENSORBOARD_DIR)
    ensure_dir(cfg.MODEL.SAVE_PATH)

    model = get_network(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model, num_gpu=cfg.MODEL.GPU_NUM)
    scheduler = make_lr_scheduler(cfg, optimizer)

    assert optimizer != None
    assert scheduler != None

    train_loader = get_train_loader(cfg)
    test_loader = get_test_loader(cfg)

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
            iter = 1
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
        # print(f'current acc: {acc}')
        print("==================================================")
        tb_writer.add_scalar("loss", avg_loss, global_step=current_epoch)
        tb_writer.add_scalar("acc", acc, global_step=current_epoch)
        if acc >= best_acc:
            best_acc = acc
            best_epoch = current_epoch
            torch.save(model.module.state_dict(), Path(cfg.MODEL.BEST_PATH))

        iter = 1       # reset iter ..
        model.train()  # open train mode ..
    print('training process finish !!!')
    print(f"best acc: {best_acc}, best epoch: {best_epoch}")

def set_seed(seed=cfg.SEED_NUM):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

if __name__ == "__main__":
    if cfg.SET_SEED:
        set_seed()
    main()