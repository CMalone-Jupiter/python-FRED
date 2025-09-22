import numpy as np
import os
import cv2

def read_calib_file(path):
    """
    Read KITTI-style calibration file lines like:
    parameter: <value>
    parameter: <value>
    ...
    Returns a dict mapping keys to numpy arrays.
    """
    d = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line or line.startswith('#'):
                continue
            key, vals = line.split(':', 1)
            nums = [float(x) for x in vals.strip().split()]
            d[key.strip()] = np.array(nums)
    return d

def get_corr_files(image_timestamp, lidar_dir, utm_dir):
    ############ Find closest point cloud and UTM position data ########################

    lidar_timestamps = np.array([int(filename.split('.bin')[0]) for filename in os.listdir(lidar_dir) if os.path.isfile(lidar_dir+filename)])

    closest_lidar = np.argmin(abs(lidar_timestamps-int(image_timestamp)))
    timestamp_diff = abs(int(image_timestamp)-lidar_timestamps[closest_lidar])

    timestamp_tolerance = 200000 # in microseconds (0.2 seconds)

    if timestamp_diff > timestamp_tolerance:
        raise Exception("No pointcloud in close enough proximity")
    else:
        lidar_filename = f"{lidar_dir}/{lidar_timestamps[closest_lidar]}.bin"


    utm_timestamps = np.array([int(filename.split('.txt')[0]) for filename in os.listdir(utm_dir) if os.path.isfile(utm_dir+filename)])

    closest_utm = np.argmin(abs(utm_timestamps-int(image_timestamp)))
    timestamp_diff = abs(int(image_timestamp)-utm_timestamps[closest_utm])

    if timestamp_diff > timestamp_tolerance:
        raise Exception("No utm measurment in close enough proximity")
    else:
        utm_filename = f"{utm_dir}/{utm_timestamps[closest_utm]}.txt"

    return lidar_filename, utm_filename