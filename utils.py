import os
from pathlib import Path
import numpy as np
from random import seed
from torch import manual_seed, device
from torch.cuda import manual_seed as cuda_seed, is_available
import torch.backends.cudnn as torch_cudnn
import albumentations as A
from sklearn.metrics import roc_auc_score


###############################################################
# Small hints
###############################################################
def set_seed(seed_val=0):
    np.random.seed(seed_val)
    random_state = np.random.RandomState(seed_val)
    seed(seed_val)
    manual_seed(seed_val)
    cuda_seed(seed_val)
    torch_cudnn.deterministic = True
    torch_cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    return random_state


def check_and_create_folder(folder):
    folder = Path(folder)
    if not folder.is_dir():
        Path.mkdir(folder)


def get_device():
    if is_available():
        now_device = device("cuda")
    else:
        now_device = device('cpu')
    return now_device


########################################################
# Work with images
########################################################
def get_train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=.5),
            A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.01, scale_limit=0.05, p=.25),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.augmentations.transforms.GaussNoise(var_limit=(10., 60.), p=0.2),
            A.augmentations.transforms.FancyPCA(alpha=10, p=0.3),
            A.augmentations.transforms.ChannelShuffle(p=0.3),
        ]
    )


###########################################################
# Metrics
###########################################################
def get_roc_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score


