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

fig, ax = plt.subplots(figsize=(12.8, 8))
# idx = [0]  # mutable index
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

        # filtered_points = pointcloud.points[(semantic_labels==0) & (abs(pointcloud.points[:,1]) < 1),:]
        # max_lookahead = filtered_points[:,0].max()
        # far_points = filtered_points[filtered_points[:,0]==max_lookahead,:]

        # if far_points.shape[0] > 1:
        #     far_point = far_points[abs(far_points[:,1]) == abs(far_points[:,1]).min(),:]
        # else:
        #     far_point = far_points
        # far_point_cam, far_point_distnace, far_point_intensity = pointcloud.select_points_ouster_to_cam(far_point)
        # far_pixel = image.get_image_coords(far_point_cam)

        # if far_pixel is not None and len(far_pixel) > 0:
        #     u, v = far_pixel[0]  # pixel coordinates

        # h, w = img_vis.shape[:2]
        # bottom_center = (w // 2, h)

        # ax.plot(
        #     [bottom_center[0], u],
        #     [bottom_center[1], v],
        #     color="lime",
        #     linewidth=2
        # )
        # ax.text(
        #     u,
        #     v - 10,
        #     f"{far_point[0,0]:.2f}",
        #     color="lime",
        #     fontsize=12,
        #     ha="center",
        #     bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
        # )

        ax.imshow(img_vis[:, :, ::-1])
        ax.set_title(f"{image_timestamp}.png")
        ax.axis("off")
        # plt.savefig('paper_figures/labelled_pointcloud.pdf', format="pdf", bbox_inches='tight')
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