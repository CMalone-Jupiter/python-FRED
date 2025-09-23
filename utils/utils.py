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

def get_corr_files(image_timestamp, dirs, tol=200000):
    output_filenames = []
    for dir in dirs:
        ############ Find closest point cloud and UTM position data ########################
        filenames = np.array([filename for filename in os.listdir(dir) if os.path.isfile(dir+filename)])
        timestamps = np.array([int(filename.split('.')[0]) for filename in os.listdir(dir) if os.path.isfile(dir+filename)])

        closest_lidar = np.argmin(abs(timestamps-int(image_timestamp)))
        timestamp_diff = abs(int(image_timestamp)-timestamps[closest_lidar])

        timestamp_tolerance = tol # in microseconds (0.2 seconds)

        if timestamp_diff > timestamp_tolerance:
            raise Exception(f"No timestamp in {dir} in close enough proximity to {image_timestamp}")
        else:
            output_filenames.append(f"{dir}/{filenames[closest_lidar]}")

    if len(dirs) == 1:
        return output_filenames[0]
    else:
        return tuple(output_filenames)



    # utm_timestamps = np.array([int(filename.split('.txt')[0]) for filename in os.listdir(dir_two) if os.path.isfile(dir_two+filename)])

    # closest_utm = np.argmin(abs(utm_timestamps-int(image_timestamp)))
    # timestamp_diff = abs(int(image_timestamp)-utm_timestamps[closest_utm])

    # if timestamp_diff > timestamp_tolerance:
    #     raise Exception("No utm measurment in close enough proximity")
    # else:
    #     utm_filename = f"{dir_two}/{utm_timestamps[closest_utm]}.txt"

    # return lidar_filename, utm_filename