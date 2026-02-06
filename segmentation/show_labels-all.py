import os
import argparse
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
import json
import yaml  # pip install pyyaml

cmap = plt.get_cmap("jet")

# ---------------- Argument Parsing ---------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Path to config file (YAML or JSON). If provided, overrides other args.")

parser.add_argument("--location", type=str, default="Cambogan", help="Location name (e.g., Cambogan)")
parser.add_argument("--sequence", type=str, default="20250811_113017", help="Sequence ID (e.g., 20250811_113017)")
parser.add_argument("--condition", type=str, default="flooded", help="Condition (e.g., flooded)")
parser.add_argument("--camera_pos", type=str, default="front", help="Camera position (e.g., front)")
parser.add_argument("--root", type=str, default="../Datasets/FRED/", help="Root dataset directory (e.g., ../Datasets/FRED/)")
parser.add_argument("--img_calib_file", type=str, default="./camera_calib.txt", help="Path to camera calibration file (e.g., ./camera_calib.txt)")

args = parser.parse_args()

# ---------------- Config Loading ---------------- #
if args.config:
    if args.config.endswith(".yaml") or args.config.endswith(".yml"):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    elif args.config.endswith(".json"):
        with open(args.config, "r") as f:
            cfg = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")

    location = cfg["location"]
    sequence = cfg["sequence"]
    condition = cfg["condition"]
    camera_pos = cfg["camera_pos"]
    root = cfg["root"]
    root_directory = f"{root}/{condition}/KITTI-style"
    img_calib_file = cfg["img_calib_file"]

else:
    # Fallback: require all CLI args
    required_args = ["location", "sequence", "condition", "camera_pos", "root", "img_calib_file"]
    missing = [arg for arg in required_args if getattr(args, arg) is None]
    if missing:
        parser.error(f"Missing arguments: {', '.join(missing)} (or provide --config)")

    location = args.location
    sequence = args.sequence
    condition = args.condition
    camera_pos = args.camera_pos
    root_directory = f"{args.root}/{args.condition}/KITTI-style"
    img_calib_file = args.img_calib_file

# # User parameters
# # location = 'Cambogan'
# # sequence = '20250811_113017'
# # location = 'Holmview'
# # sequence = '20250820_130327'
# # location = 'Pullenvale'
# # sequence = '20250916_124105'
# location = 'Mount-Cotton'
# sequence = '20241217_113410'
# condition = 'flooded'
# camera_pos = 'front'
# root_directory = f"../Datasets/FRED/{condition}/KITTI-style"
# # 01000000

############ Define filenames and directories ####################################

image_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-imgs/"
label_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-labels/"
img_calib_file = f"./camera_calib.txt"
timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(image_dir)) if os.path.isfile(image_dir+filename)]

fig, ax = plt.subplots(figsize=(12.8, 8))
# idx = [0]  # mutable index
idx = [183]  # mutable index

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

        # Catch for when there is no label or all pixels are labelled as 'other' (i.e. not labelled) #
        if not np.all(image.colour_label == image.semantic_classes['other']):
            overlay_img[label_mask] = cv2.addWeighted(image.image[label_mask], 0.5, image.colour_label[label_mask], 0.5, 0)

        ax.imshow(overlay_img[:, :, ::-1])
        ax.set_title(f"{image_timestamp}.png")
        ax.axis("off")
        # plt.savefig('paper_figures/semantic_image_labels.pdf', format="pdf", bbox_inches='tight')
        fig.canvas.draw()
    except Exception as e:
        print(f"Could not show label for {image_timestamp}.png: {e}")
        idx[0] += 1
        show_image(idx[0])  # skip bad one

def on_key(event):
    if event.key in [' ', 'right']:  # space or right arrow
        idx[0] += 1
        show_image(idx[0])
    elif event.key in [' ', 'left']:  # space or right arrow
        if idx[0] > 0:
            idx[0] -= 1
            show_image(idx[0])
    elif event.key in ['q', 'escape']:  # q or Esc → quit
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
show_image(idx[0])
plt.show()