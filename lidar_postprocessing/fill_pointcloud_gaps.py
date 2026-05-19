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
LABEL_UNKNOWN = -1

# User parameters
location = 'Cambogan'
sequence = '20250811_113017'
# location = 'Holmview'
# sequence = '20250820_130327'
# location = 'Mount-Cotton'
# sequence = '20241217_113410'
condition = 'flooded'
camera_pos = 'front'
root_directory = f"C:/Users/conno/Documents/data/FRED/{condition}/KITTI-style/" #f"../Datasets/FRED/{condition}/KITTI-style"
# 01000000

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

# timestamps.sort()

fig, ax = plt.subplots(figsize=(12.8, 8))

# idx = [0]  # mutable index
# idx = [0]
idx = [160]

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
        # print(f"Number of points in pointcloud: {pointcloud.points.shape}")
        # # print(f"Nan's in pointcloud? {np.any( == np.nan)}")
        # print(f"Number of zeroed points: {np.sum(np.all(pointcloud.points == 0, axis=1))}")
        # print(f"invalid points in ground plane: {np.sum(pointcloud.ground_semantic[np.all(pointcloud.points == 0, axis=1)]==0)}")
        pointcloud.points, pointcloud.ground_semantic, pointcloud.ground_inlier = pointcloud.destagger() #pointcloud.points, pointcloud.ground_semantic, pointcloud.ground_inlier
        groundplane_eqn = utils.fit_height_field_linear(pointcloud.points[pointcloud.ground_semantic==0,:3])
        pointcloud.points, interp_flags = utils.complete_cloud(pointcloud.points, groundplane_eqn)

        pointcloud.points = pointcloud.points[interp_flags]

        point_cam, distances_cam, intensities_cam, all_points_cam, valid_cam = pointcloud.points_ouster_to_cam() #, beam_id, azimuth
        img_vis, uv, valid_img, _ = image.project_points(all_points_cam, intensities_cam, cmap, valid_cam, colour_norm=255) #, beam_id, azimuth
        # semantic_labels = interp_flags.astype(int) + 1

        # labels_norm = semantic_labels.astype(np.float64) / semantic_labels.max()

        # colors = np.stack(
        #     (labels_norm, np.zeros(labels_norm.shape[0]), np.zeros(labels_norm.shape[0])),
        #     axis=1
        # )  # shape (N, 3)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pointcloud.points[:,:3])

        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd,])
        ax.imshow(img_vis[:,:,::-1])
        ax.set_title(f"{i+1}/{len(timestamps)} — {image_timestamp}.png\n(close window or press any key to continue)")
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

# while idx[0] < len(timestamps):
fig.canvas.mpl_connect('key_press_event', on_key)
show_image(idx[0])
plt.show()
print(f"Finished all pointclouds")