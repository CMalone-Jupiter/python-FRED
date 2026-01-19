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
root_directory = f"../Datasets/FRED/{condition}/KITTI-style"
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

# idx = [0]  # mutable index
idx = [0]

def show_image(i):
    # ax.clear()
    if i >= len(timestamps):
        return
    image_timestamp = timestamps[i]
    try:
        image_filename = f"{image_dir}/{image_timestamp}.png"
        label_filename = f"{label_dir}/{image_timestamp}.png"
        lidar_filename, utm_filename = utils.get_corr_files(image_timestamp, [lidar_dir, utm_dir])

        image = ImageData(image_filename, img_calib_file, label_filename)
        pointcloud = PointCloud(lidar_filename, lidar_calib_file)
        groundplane_eqn = utils.fit_height_field_linear(pointcloud.points[pointcloud.ground_semantic==0,:3])
        pointcloud.points, interp_flags = utils.complete_cloud(pointcloud.points, groundplane_eqn)

        point_cam, distances_cam, intensities_cam, all_points_cam, valid_cam = pointcloud.points_ouster_to_cam() #, beam_id, azimuth
        img_vis, uv, valid_img = image.project_points(all_points_cam, intensities_cam, cmap, valid_cam, colour_norm=255) #, beam_id, azimuth
        semantic_labels = interp_flags.astype(int) + 1

        labels_norm = semantic_labels.astype(np.float64) / semantic_labels.max()

        colors = np.stack(
            (labels_norm, np.zeros(labels_norm.shape[0]), np.zeros(labels_norm.shape[0])),
            axis=1
        )  # shape (N, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud.points[:,:3])

        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd,])

        idx[0] += 1


    except Exception as e:
        print(f"Could not project pointcloud onto {image_timestamp}.png: {e}")
        idx[0] += 1
        show_image(idx[0])  # skip bad one


while idx[0] < len(timestamps):
    show_image(idx[0])
print(f"Finished all pointclouds")