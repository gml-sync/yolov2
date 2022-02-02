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
    coco_files = sorted(Path(settings.output_path).rglob("*.jpg"))
    # 5000 files in each folder

    for idx in range(len(gt_files)):
        gt = cv2.imread(str(gt_files[idx]))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        h, w = gt.shape

        # count variance by row (top/bottom strips)
        # var = avg( (x_i - avg(x))^2 )
        # variance works very bad!
        sobel = gt[:, 1:] - gt[:, :w-1]
        variance = np.average(sobel ** 2, axis=1)
        high_var = variance > 1e-5
        grid = np.arange(h)
        min_w = grid[high_var].min()
        max_w = grid[high_var].max() + 1
        if np.sum(~high_var[min_w:max_w]) != 0:
            print("Strip detection error on image", idx)
        # print(f"image {str(gt_files[idx])} min {min_w} max {max_w} minvar {np.min(variance)} maxvar {np.max(variance)}")
        # cv2.imwrite(f"{idx:05d}_cut.jpg", np.clip(gt * 255, 0, 255).astype(np.uint8),
        #             [cv2.IMWRITE_JPEG_QUALITY, 100])

        # count variance by column (left/right strips)
        sobel = gt[1:, :] - gt[:h - 1, :]
        variance = np.average(sobel ** 2, axis=0)
        high_var = variance > 1e-5
        grid = np.arange(w)
        min_h = grid[high_var].min()
        max_h = grid[high_var].max() + 1
        if np.sum(~high_var[min_h:max_h]) != 0:
            print("Strip detection error on image", idx)

        print(gt[:5, :5])


        if idx > 1:
            break


class Settings:
    def __init__(self):
        self.gt_images_path = "visualize"
        self.output_path = "outputs-0"
        self.coco_path = "../datasets/coco5k_ref/images"

def main():
    settings = Settings()
    cut_and_save(settings, None)
