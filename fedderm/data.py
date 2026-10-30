"""Fed-ISIC2019 data, with FLamby's exact augmentations (commit edacf54)."""
import os
import random

import albumentations
import numpy as np
import pandas as pd
import torch
from PIL import Image

SZ = 200
NUM_CLIENTS = 6
NUM_CLASSES = 8


def flamby_train_aug():
    # identical to flamby/datasets/fed_isic2019/dataset.py (train=True)
    return albumentations.Compose([
        albumentations.RandomScale(0.07),
        albumentations.Rotate(50),
        albumentations.RandomBrightnessContrast(0.15, 0.1),
        albumentations.Flip(p=0.5),
        albumentations.Affine(shear=0.1),
        albumentations.RandomCrop(SZ, SZ),
        albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
        albumentations.Normalize(always_apply=True),
    ])


def flamby_test_aug():
    return albumentations.Compose([
        albumentations.CenterCrop(SZ, SZ),
        albumentations.Normalize(always_apply=True),
    ])


class IsicDataset(torch.utils.data.Dataset):
    """Images of one center (or all centers) and one fold of a split CSV.

    split CSV columns: image, center, target, lesion_id, fold (train / test / val)."""

    def __init__(self, split_csv, img_dir, fold, center=None, train_aug=False, max_items=None):
        df = pd.read_csv(split_csv)
        df = df[df.fold == fold]
        if center is not None:
            df = df[df.center == center]
        df = df.reset_index(drop=True)
        if max_items:
            df = df.groupby("center", group_keys=False).head(max_items).reset_index(drop=True)
        self.df = df
        self.paths = [os.path.join(img_dir, f"{n}.jpg") for n in df.image]
        self.targets = df.target.to_numpy()
        self.aug = flamby_train_aug() if train_aug else flamby_test_aug()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]))
        img = self.aug(image=img)["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(img), torch.tensor(int(self.targets[i]), dtype=torch.int64)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, shuffle, batch_size, workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        persistent_workers=(workers > 0 and shuffle), worker_init_fn=seed_worker, generator=g,
        pin_memory=torch.cuda.is_available())


class InfiniteLoader:
    """FLamby's DataLoaderWithMemory: restarts the iterator when exhausted."""

    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)
        self.n = len(loader.dataset)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)
