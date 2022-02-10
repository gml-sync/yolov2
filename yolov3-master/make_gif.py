from glob import glob
from pathlib import Path
import os

import numpy as np
import cv2
import imageio
from skimage import io
import matplotlib.pyplot as plt

new_path = "d:/ra-examples/266"
#old_path = "d:/ra-examples/old_model"
output_path = "d:/example.png"

# ------------------------------------
#    Method 1 (gif): imageio.mimwrite
# ------------------------------------
# files = ["1_gt_image", "1_pred", "2_pred"]
# image_list = []
# for path in files:
#     filename = os.path.join(new_path, path + ".jpg")
#     image = io.imread(filename)
#     image_list.append(image)
# imageio.mimwrite(output_path, image_list, "gif", duration=10)

# ------------------------------------
#    Method 2 (mp4): imageio - ffmpeg video
# ------------------------------------
# files = ["1_gt_image", "1_pred", "2_pred"]
# writer = imageio.get_writer('test.mp4', fps=30)
# for path in files:
#     filename = os.path.join(new_path, path + ".jpg")
#     image = io.imread(filename)
#     for i in range(60):
#         writer.append_data(image)
# writer.close()


#    Method 3: create APNG (doesn't show animation on Windows, but works in Google Docs)

#files = ["example_20", "example_37", "example_42", "example_47", "example_52"]
#files = [os.path.join(new_path, filename + ".bmp") for filename in files]
files = sorted(os.listdir(new_path))
files = [os.path.join(new_path, filename) for filename in files]
APNG.from_files(files, delay=1000).save(output_path)
