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
location = 'Cambogan'
sequence = '20250811_113017'
# location = 'Holmview'
# sequence = '20250820_130327'
# location = 'Mount-Cotton'
# sequence = '20241217_113410'
condition = 'flooded'
camera_pos = 'front'
root_directory = f"../Datasets/FRED/{condition}/KITTI-style"
# 01000000

############ Define filenames and directories ####################################

image_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-imgs/"
lidar_dir = f"{root_directory}/{location}_{sequence}/ouster/"
utm_dir = f"{root_directory}/{location}_{sequence}/utm/"

img_calib_file = f"./camera_calib.txt"
lidar_calib_file = f"./calib.txt"

timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(image_dir)) if os.path.isfile(image_dir+filename)]

# timestamps.sort()

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
        lidar_filename, utm_filename = utils.get_corr_files(image_timestamp, [lidar_dir, utm_dir])

        image = ImageData(image_filename, img_calib_file)
        pointcloud = PointCloud(lidar_filename, lidar_calib_file)

        point_cam, distances_cam, intensities_cam = pointcloud.points_ouster_to_cam()
        img_vis = image.project_points(point_cam, intensities_cam, cmap)

        ax.imshow(img_vis[:, :, ::-1])
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