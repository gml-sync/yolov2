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
    print(len(gt_files), len(out_files))
    print(gt_files[0], out_files[0])


class Settings:
    def __init__(self):
        self.gt_images_path = "visualize"
        self.output_path = "outputs-0"

settings = Settings()

cut_and_save(settings, None)