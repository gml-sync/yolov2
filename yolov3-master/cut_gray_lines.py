import sys
from glob import glob
from pathlib import Path
import os
import datetime as dt

import numpy as np
import cv2
import einops

import torch
import torch.utils.data as data
from torch import nn

def cut_and_save(settings, result_dir):
    # gt
    # ├─00000
    #   ...
    # ├─04999_0_features.png
    # ├─04999_0_image.jpg
    # └─04999_0_range.txt
    #
    # outputs
    # ├──────00000_pred.jpg
    #   ...
    # └──────04999_pred.jpg

    gt_files = sorted(Path(settings.gt_images_path).rglob("*image.jpg"))
    out_files = sorted(Path(settings.output_path).rglob("*.jpg"))
    # 5000 files in each folder

    for idx in range(len(gt_files)):
        gt = cv2.imread(gt_files[idx]).astype(np.float32) / 255
        h, w = gt.shape[:2]

        # count variance by row
        # var = avg( (x_i - avg(x))^2 )
        avg = np.average(gt, axis=1)
        variance = np.average((gt - avg) ** 2, axis=1) # broadcasting
        print(variance[:10], variance[300:310], variance[-10:])

        break


class Settings:
    def __init__(self):
        self.gt_images_path = "visualize"
        self.output_path = "outputs-0"

def main():
    settings = Settings()
    cut_and_save(settings, None)
