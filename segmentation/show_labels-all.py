import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append(os.path.abspath("."))   # one level up
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from utils.lidar import PointCloud
from utils.camera import ImageData
import utils.utils as utils
from natsort import natsorted

cmap = plt.get_cmap("jet")

# User parameters
# location = 'Cambogan'
# sequence = '20250811_113017'
location = 'Holmview'
sequence = '20250820_130327'
condition = 'flooded'
camera_pos = 'front'
root_directory = f"../Datasets/FRED/{condition}/KITTI-style"
# 01000000

############ Define filenames and directories ####################################

image_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-imgs/"
label_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-labels/"
img_calib_file = f"./camera_calib.txt"
timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(image_dir)) if os.path.isfile(image_dir+filename)]

fig, ax = plt.subplots(figsize=(12.8, 8))
idx = [0]  # mutable index

def show_image(i):
    ax.clear()
    if i >= len(timestamps):
        plt.close(fig)
        return
    image_timestamp = timestamps[i]
    try:
        image_filename = f"{image_dir}/{image_timestamp}.png"
        label_filename = f"{label_dir}/{image_timestamp}.png"
        image = ImageData(image_filename, img_calib_file, label_filename)

        label_mask = np.any(image.colour_label != image.semantic_classes['other'], axis=-1)
        overlay_img = image.image.copy()
        overlay_img[label_mask] = cv2.addWeighted(image.image[label_mask], 0.5, image.colour_label[label_mask], 0.5, 0)

        ax.imshow(overlay_img[:, :, ::-1])
        ax.set_title(f"{image_timestamp}.png")
        ax.axis("off")
        fig.canvas.draw()
    except Exception as e:
        print(f"Could not project pointcloud onto {image_timestamp}.png: {e}")
        idx[0] += 1
        show_image(idx[0])  # skip bad one

def on_key(event):
    if event.key in [' ', 'right']:  # space or right arrow
        idx[0] += 1
        show_image(idx[0])
    elif event.key in ['q', 'escape']:  # q or Esc → quit
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
show_image(idx[0])
plt.show()