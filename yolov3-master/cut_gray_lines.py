import sys
from glob import glob
from pathlib import Path
import os
import datetime as dt
import shutil
import multiprocessing as mp

import numpy as np
import cv2
import einops

import torch
import torch.utils.data as data
from torch import nn


def single_process(settings, result_dir, file_lists, idx):
    gt_files, out_files, coco_files = file_lists

    gt = cv2.imread(str(gt_files[idx]))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    out_image = cv2.imread(str(out_files[idx])).astype(np.float32) / 255
    coco_gt = cv2.imread(str(coco_files[idx])).astype(np.float32) / 255
    h, w = gt.shape

    # Count variance by row (top/bottom strips).
    # var = avg( (x_i - avg(x))^2 )
    # Variance works very bad!
    # Use sobel + difference from gray
    solid_gray = 0.44705883
    sobel_thr = 1e-5
    gray_thr = 1e-5

    sobel = gt[:, 1:] - gt[:, :w - 1]
    variance = np.average(sobel ** 2, axis=1)
    gray_diff = np.average((gt - solid_gray) ** 2, axis=1)
    # print(gray_diff)
    high_var = gray_diff > gray_thr
    grid = np.arange(h)
    min_h = grid[high_var].min()
    max_h = grid[high_var].max() + 1
    if np.sum(~high_var[min_h:max_h]) != 0:
        print("Strip detection error on image", idx)
    # print(f"image {str(gt_files[idx])} min {min_w} max {max_w} minvar {np.min(variance)} maxvar {np.max(variance)}")
    # cv2.imwrite(f"{idx:05d}_cut.jpg", np.clip(gt * 255, 0, 255).astype(np.uint8),
    #             [cv2.IMWRITE_JPEG_QUALITY, 100])

    # count variance by column (left/right strips)
    sobel = gt[1:, :] - gt[:h - 1, :]
    variance = np.average(sobel ** 2, axis=0)
    gray_diff = np.average((gt - solid_gray) ** 2, axis=0)
    high_var = gray_diff > gray_thr
    grid = np.arange(w)
    min_w = grid[high_var].min()
    max_w = grid[high_var].max() + 1
    if np.sum(~high_var[min_w:max_w]) != 0:
        print("Strip detection error on image", idx)

    res_h, res_w = max_h - min_h, max_w - min_w
    coco_h, coco_w = coco_gt.shape[:2]
    if (res_h / res_w - coco_h / coco_w) ** 2 > 0.001:
        print("Aspect ratio violation on image", idx, "coco", str(coco_files[idx]))
    print(f"idx: {idx} h: {res_h} w: {res_w}")
    return

    res_image = out_image[min_h:max_h, min_w:max_w]
    dim = (coco_w, coco_h)
    resized = cv2.resize(res_image, dim)

    res_path = str(result_dir / f"{idx:05d}_cut.jpg")
    cv2.imwrite(str(result_dir / coco_files[idx].name), np.clip(resized * 255, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    res_h, res_w = resized.shape[:2]
    #print(f"image {str(gt_files[idx])} res_h {res_h} res_w {res_w} coco_h {coco_h} coco_w {coco_w} aspect diff {res_h / res_w - coco_h / coco_w}")

    if idx % 500 == 0:
        print(idx)

    return None

def cut_and_save(settings):
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
    coco_files = sorted(Path(settings.coco_path).rglob("*.jpg"))
    # 5000 files in each folder

    for params_idx in range(len(settings.output_paths)):
        output_path = settings.output_paths[params_idx]
        result_dir = Path(settings.result_paths[params_idx])
        out_files = sorted(Path(output_path).rglob("*.jpg"))
        file_lists = [gt_files, out_files, coco_files]
        for idx in range(len(gt_files)):
            single_process(settings, result_dir, file_lists, idx)

    # # Step 1: Init multiprocessing.Pool()
    # pool = mp.Pool(10)
    # # Step 2: pool.apply
    # results = [pool.apply_async(single_process, args=(settings, result_dir, file_lists, idx)) for idx in range(len(gt_files))]
    # # Step 3: Don't forget to close
    # pool.close()
    # pool.join()

class Settings:
    def __init__(self):
        self.gt_images_path = "visualize"
        self.output_paths = ["outputs-0", "outputs-27", "outputs-37"]
        self.result_paths = ["cut-0", "cut-27", "cut-37"]
        self.coco_path = "../datasets/coco5k_ref/images"

def main():
    settings = Settings()

    for result_dir in settings.result_paths:
        Path(result_dir).mkdir(exist_ok=True)

    cut_and_save(settings)
