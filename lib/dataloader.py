from locale import normalize
from random import shuffle
from tkinter.filedialog import test
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset.dataset import Dataset


def get_train_loader(cfg, is_shuffle = True, use_augmentation = True):
    # transform img
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    DatasetCreator = Dataset

    dataset = DatasetCreator(cfg, 'train', transform, use_augmentation)
    is_shuffle = cfg.DATALOADER.SHUFFLE
    train_loader = DataLoader(dataset, shuffle=is_shuffle, batch_size=cfg.DATALOADER.BATCH_SIZE, \
                             num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True)

    return train_loader

def get_test_loader(cfg, is_shuffle=False, use_augmentation=False):
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    DatasetCreator = Dataset

    dataset = DatasetCreator(cfg, 'test', transform, use_augmentation)

    test_loader = DataLoader(dataset, shuffle=is_shuffle, batch_size=1, num_workers=1)

    return test_loader