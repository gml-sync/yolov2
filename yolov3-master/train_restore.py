from glob import glob
from pathlib import Path

import numpy as np
import cv2
import einops

import torch
import torch.utils.data as data
from torch import nn
from torch.autograd import Variable

DEVICE = 'cuda'

class RestorationDataset(data.Dataset):
    def __init__(self):
        self.feature_list = [] # input
        self.image_list = [] # gt
        self.desc_list = [] # input range - min, max

        dataset_root = Path("visualize")

        self.feature_list = sorted(glob(dataset_root / "*features.png"))
        self.image_list = sorted(glob(dataset_root / "*image.jpg"))
        self.desc_list = sorted(glob(dataset_root / "*range.txt"))

    def __getitem__(self, index):
        index = index % len(self.image_list)

        min_feat, max_feat = 0, 0
        with open(self.desc_list[index], "r") as description:
            min_feat, max_feat = map(float, description.read().split())
        features = cv2.imread(self.feature_list[index]).astype(np.float32)
        features = features / 255 * (max_feat - min_feat) + min_feat
        features = features[:, :, 0]
        features = einops.rearrange(features, '(i1 h) (i2 w) -> (i1 i2) h w', h=80, w=80) # 80 is specific to layer 5
        image = cv2.imread(self.image_list[index]).astype(np.float32)

        features = torch.from_numpy(features).float()
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return features, image
        
    def __len__(self):
        return len(self.image_list)

train_dataset = RestorationDataset()
f, i = train_dataset[0]
print(f.shape, i.shape)
print(train_dataset.feature_list[:5],
    train_dataset.image_list[:5],
    train_dataset.desc_list[:5])

