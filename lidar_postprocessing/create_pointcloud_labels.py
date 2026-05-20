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

hf_app = True

if hf_app:
    from huggingface_hub import snapshot_download

cmap = plt.get_cmap("jet")
LABEL_UNKNOWN = -1

# User parameters
location = 'Cambogan'
sequence = '20250811_113017'
condition = 'flooded'
camera_pos = 'front'
root_directory = f"/data/FRED/{condition}/KITTI-style"

if (not os.path.exists(root_directory)) and (hf_app):
    snapshot_download(
        repo_id="CMalone-Jupiter/FRED",
        repo_type="dataset",
        local_dir="/data/FRED",
        allow_patterns=f"{condition}/KITTI-style/{location}_{sequence}/**",
        token=os.environ.get("HF_TOKEN")
    )

############ Define filenames and directories ####################################

image_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-imgs/"
label_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-labels/"
lidar_dir = f"{root_directory}/{location}_{sequence}/ouster/"
utm_dir = f"{root_directory}/{location}_{sequence}/utm/"

img_calib_file = f"./camera_calib.txt"
lidar_calib_file = f"./calib.txt"

timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(image_dir)) if os.path.isfile(image_dir+filename)]
groundplane_eqn = tuple(np.loadtxt(f"{root_directory}/{location}_{sequence}/ground_plane_eqn.txt"))
a, b, c, d = groundplane_eqn


fig, ax = plt.subplots(figsize=(12.8, 8))
idx = [183]

def show_image(i):
    ax.clear()
    if i >= len(timestamps):
        plt.close(fig)
        return
    image_timestamp = timestamps[i]
    try:
        image_filename = f"{image_dir}/{image_timestamp}.png"
        label_filename = f"{label_dir}/{image_timestamp}.png"
        lidar_filename, utm_filename = utils.get_corr_files(image_timestamp, [lidar_dir, utm_dir])

        image = ImageData(image_filename, img_calib_file, label_filename)
        pointcloud = PointCloud(lidar_filename, lidar_calib_file)

        point_cam, distances_cam, intensities_cam, all_points_cam, valid_cam = pointcloud.points_ouster_to_cam() #, beam_id, azimuth
        img_vis, uv, valid_img = image.project_points(all_points_cam, intensities_cam, cmap, valid_cam) #, beam_id, azimuth

        valid_semantic = valid_cam & valid_img

        # Assign semantics
        semantic_labels = utils.assign_semantic_labels(
            pointcloud.points[:, :3],
            uv,
            valid_semantic,
            image.label_img,
            interp_flags=None,
            unknown_label=3
        )

        print(f"Max x distance: {pointcloud.points[semantic_labels==0,0].max()}")

        ground_filter = pointcloud.ground_semantic == 0
        inlier_filter = pointcloud.ground_inlier == 1

        img_vis, uv, valid_img = image.project_points(all_points_cam, semantic_labels, cmap, valid_cam) #, beam_id, azimuth

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
    elif event.key in [' ', 'left']:  # space or right arrow
        if idx[0] > 0:
            idx[0] -= 1
            show_image(idx[0])
    elif event.key in ['q', 'escape']:  # q or Esc → quit
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
show_image(idx[0])
plt.show()