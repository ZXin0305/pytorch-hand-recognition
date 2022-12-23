import torch.optim as optim
import torch
from torch.optim import lr_scheduler

def make_lr_scheduler(cfg, optimizer):
    scheduler = None
    if cfg.SOLVER.SCHEDULER == "cosin":
        lr_lambda = lambda epoch : (cfg.END_EPOCH - epoch) / cfg.END_EPOCH
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif cfg.SOLVER.SCHEDULER == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.DROP_STEP, gamma=cfg.SOLVER.GAMMA)
    elif cfg.SOLVER.SCHEDULER == "samplestep":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.STEP_SIZE, gamma=cfg.SOLVER.GAMMA)
    return scheduler

def make_optimizer(cfg, model, num_gpu):
    optimizer = None
    if cfg.SOLVER.NAME == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                              weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.NAME == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999), eps=1e-08, 
                               weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    
    return optimizer