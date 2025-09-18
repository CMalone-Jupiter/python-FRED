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

# def read_calib_file(self, path):
#     """
#     Read camera intrinsics in file with format:
#     focal_len: <value> The camera focal length in mm
#     principal_x: <value> principal x coordinate in pixels
#     principal_y: <value> principal y coordinate in pixels
#     pp_mm_x: <value> pixels per mm in x direction
#     pp_mm_y: <value> pixels per mm in y direction
#     Returns a dict mapping keys to numpy arrays.
#     """
#     d = {}
#     with open(path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or ':' not in line or line.startswith('#'):
#                 continue
#             key, vals = line.split(':', 1)
#             nums = [float(x) for x in vals.strip().split()]
#             d[key.strip()] = np.array(nums)
#     return d