from easydict import EasyDict as edict
from IPython import embed
from path import Path
import os

class Config:
    # --------- Directory ---------- #
    HOME = "/home/xuchengjun/ZXin/pytorch-hand-recognition"
    LOG_DIR = "/home/xuchengjun/ZXin/pytorch-hand-recognition/log"
    LOG_FILE_NAME = "log_20221214.txt"
    DATA = ""  #日期
    TENSORBOARD_DIR = os.path.join(LOG_DIR, DATA)

    # --------- Dataloader ---------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 1
    DATALOADER.BATCH_SIZE = 6
    DATALOADER.SHUFFLE = True

    # --------- Dataset --------- #
    DATASET = edict()
    # /media/xuchengjun/disk1/zx/left  right
    DATASET.ROOT_PATH = "/media/xuchengjun/disk1/zx/right"   # train  直到/HAND的上一级   记得改这个
    # /media/xuchengjun/disk1/zx/left/other_rate/left_train(1:9).json
    DATASET.PATH = "/media/xuchengjun/disk1/zx/right/right_train.json"  # new_train
    DATASET.SUFFIX = ['jpg', 'png', 'csv']   # rgb, depth, skel
    DATASET.AUG_NAME = "aug_data"

    # --------- Input --------- #
    INPUT = edict()
    INPUT.NORMALIZE = True
    INPUT.MEANS = [0.406, 0.456, 0.485]  # 仅仅针对于RGB图像
    INPUT.STDS = [0.225, 0.224, 0.229]
    INPUT.SHAPE = (160, 160)
    # INPUT.BATCH_SIZE = 12
    INPUT.CHANEL_NUM = 5  # 如果是双通道的话，就这样， 否则就直接是5

    # --------- Output -------- #
    OUTPUT = edict()
    """
    right, left
    """
    OUTPUT.CLASS_NUM = 11

    # --------- Model -------- #
    MODEL = edict()
    """
    attension, densenet, googlenet, inceptionv3, inceptionv4, mobilenet, mobilenetv2
    resnet, resnext, shufflenet, shufflenetv2, vgg, << mynet >>
    VSGCNN LSTM  resnet101
    """
    MODEL.NAME = "mynet"
    MODEL.DEVICE = "cuda"
    MODEL.USE_GPU = True
    MODEL.GPU_IDS = [0, 1, 2]
    MODEL.GPU_NUM = len(MODEL.GPU_IDS)
    MODEL.SAVE_PATH = "/media/xuchengjun/disk1/zx/model/20221214"   # 每次这个记住改!
    MODEL.SAVE_PERIOD = 1   #每多少个epoch进行保存
    MODEL.CHECKPOINT_PATH = "/media/xuchengjun/disk1/zx/model/20221214"
    MODEL.CHECK_PERIOD = 500 # 每多少个iter进行保存
    MODEL.WEIGHT = "/media/xuchengjun/disk1/zx/model/20221214/checkpoint.pth"   # 这个是为了断点的时候导入之前的权重
    MODEL.BEST_PATH = "/media/xuchengjun/disk1/zx/model/20221214/best.pth"
    MODEL.TEST_PATH = "/media/xuchengjun/disk1/zx/model/right/right_multi.pth"
    MODEL.DROPOUT_RATE = [0.2, 0.2] # FC的Dropout层
    MODEL.DP = True
    MODEL.INIT_WEIGHTS = True

    # --------- Training Cfg ----------- #
    SOLVER = edict()
    SOLVER.BASE_LR = 1e-3
    SOLVER.WEIGHT_DECAY = 8e-6
    SOLVER.MOMENTUM = 0.9
    """
    SGD, ADAM,...
    """
    SOLVER.NAME = "sgd"
    SOLVER.SCHEDULER = "multistep"  # cosin, multistep, samplestep
    SOLVER.DROP_STEP = [5, 10, 15, 20]  # 10, 20, 30, 45   //  5, 10, 15, 20
    SOLVER.GAMMA = 0.1
    SOLVER.STEP_SIZE = 10

    START_EPOCH = 1
    END_EPOCH = 24

    # -------- Loss ------------ #
    LOSS = edict()

    # -------- Test ---------- #
    TEST = edict()
    TEST.ROOT_PATH = "/media/xuchengjun/disk1/zx/HAND"
    # /media/xuchengjun/disk1/zx/left/other_rate/left_test(1:9).json
    # /media/xuchengjun/disk1/zx/left/other_rate/left_test(3:7).json
    TEST.PATH = "/media/xuchengjun/disk1/zx/right/right_test.json"

    RUN_EFFICIENT = False 
    SET_SEED = False
    SEED_NUM = 47
    AUGMENTATION_PROB1 = 0.98 # HSV 0.95
    AUGMENTATION_PROB2 = 0.9   # Gray  0.7
    AUGMENTATION_PROB3 = 0.5  # Rotate 0.3
    AUGMENTATION_PROB4 = 0.8  # gridmask  0.9 
    AUGMENTATION_PROB5 = 0.8 # mask center 0.9
    MAX_ROTATE_DEGREE = 30

    # AUGMENTATION_PROB1 = 1 # HSV 0.95
    # AUGMENTATION_PROB2 = 1   # Gray  0.7
    # AUGMENTATION_PROB3 = 1  # Rotate 0.3
    # AUGMENTATION_PROB4 = 1  # gridmask  0.9 
    # AUGMENTATION_PROB5 = 0.0 # mask center 0.9
    # MAX_ROTATE_DEGREE = 30

config = Config()
cfg = config