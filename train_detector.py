#!/usr/bin/env python
"""
Object detection training script 
"""

import argparse
from ast import literal_eval
import gc
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
import time
import random

# Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# torch
import torch
import torchvision
from torch.utils.data import DataLoader

from src.data.datasets import DetectionDataset
from src.models.detection import DetectionBaseline
from src.train.DetectionTrainer import DetectionTrainer


class Config:
    """ Configuration for the training """

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_name = f"classification_{time_stamp}"
    log_path = Path("./logs") / exp_name
    fold_num: int = 0
    seed: int = 2021
    num_classes: int = 2
    n_anchors: int = 1  # number of anchors per pixel location
    freeze_backbone = True  # don't train backbone weights if true
    img_size: int = (768,) * 2  # implementation expects height = width
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')  # computing device
    num_workers: int = 8  # number of processors used to prepare the batches
    batch_size: int = 2
    n_epochs: int = 2  # 100
    lr: float = 1e-3  # 0.001
    use_scheduler = False
    label_dict = {
        0: "negative",
        1: "positive"
    }


def main(args):
    # Setup
    seed_everything(Config.seed)
    start = time.time()

    # Load data
    data_path = Path(args.data_path)
    train_path = data_path / "train"

    ann_df = pd.read_csv(data_path / "train_annotations.csv", converters={
        "boxes": literal_eval,
        "labels": literal_eval,
        "pixel_spacing": literal_eval
    })

    # Image Pre-processing
    # normalization transforms
    norm_transform_list = [
        A.Resize(height=Config.img_size[0], width=Config.img_size[1], p=1.0),
        A.Normalize(mean=(0,), std=(1,), p=1.0),
        ToTensorV2(p=1.0)
    ]

    # augmentation transforms
    aug_transform_list = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.CropAndPad(percent=(0.0, 0.02), pad_mode=cv2.BORDER_CONSTANT,
                     pad_cval=0, keep_size=True, sample_independently=True, p=0.5)
    ]
    # bounding box format
    bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']}

    train_transforms = A.Compose(
        aug_transform_list + norm_transform_list, bbox_params=bbox_params)
    val_transforms = A.Compose(norm_transform_list, bbox_params=bbox_params)

    # Datasets
    train_df = ann_df[ann_df['fold'] != Config.fold_num]
    val_df = ann_df[ann_df['fold'] == Config.fold_num]

    train_ds = DetectionDataset(train_df, train_path, train_transforms)
    val_ds = DetectionDataset(val_df, train_path, val_transforms)

    # Dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn
    )

    # Model
    model = get_model(Config)
    model.to(Config.device)

    # Training
    trainer = DetectionTrainer(model, Config)
    trainer.fit(train_dl, val_dl)

    duration = time.time() - start
    print(f"Training done. Duration: {duration:.2f} s")


# Helper Functions

def seed_everything(seed):
    """ seed random number generators to make runs deterministic """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(cfg, checkpoint_path=None):
    model = DetectionBaseline(cfg)

    # Load the trained weights
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        del checkpoint
        gc.collect()

    return model.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./data/siim-covid19-detection-subset",
                        metavar="PATH", help="path to root of train data")

    args = parser.parse_args()
    main(args)
