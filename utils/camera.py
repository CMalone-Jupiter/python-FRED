import numpy as np
import cv2
from PIL import Image

class ImageData():
    def __init__(self, file_path, calib_path):
        self.image = cv2.imread(file_path)

        intrinsics = self.read_calib_file(calib_path)
        focal_len = intrinsics['focal_len']
        principal_x = intrinsics['principal_x']
        principal_y = intrinsics['principal_y']
        pp_mm_x = intrinsics['pp_mm_x']
        pp_mm_y = intrinsics['pp_mm_y']

        self.camera_matrix = self.create_camera_matrix(focal_len, principal_x, principal_y, 
                                                       pp_mm_x, pp_mm_y)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    def read_calib_file(self, path):
        """
        Read camera intrinsics in file with format:
        focal_len: <value> The camera focal length in mm
        principal_x: <value> principal x coordinate in pixels
        principal_y: <value> principal y coordinate in pixels
        pp_mm_x: <value> pixels per mm in x direction
        pp_mm_y: <value> pixels per mm in y direction
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

    def create_camera_matrix(focal_length_mm, principal_point_x_pixels, principal_point_y_pixels,
                            pixels_per_mm_x, pixels_per_mm_y):
        """
        Create camera matrix from physical camera parameters.
    
        Args:
            focal_length_mm: Focal length in millimeters
            principal_point_x_pixels: Principal point X coordinate in pixels
            principal_point_y_pixels: Principal point Y coordinate in pixels
            pixels_per_mm_x: Pixels per millimeter in X direction
            pixels_per_mm_y: Pixels per millimeter in Y direction
        
        Returns:
            camera_matrix: 3x3 camera intrinsic matrix
        """
    
        # Check for None or invalid values
        if any(x is None for x in [focal_length_mm, principal_point_x_pixels, principal_point_y_pixels,
                                pixels_per_mm_x, pixels_per_mm_y]):
            print("ERROR: One or more input parameters is None!")
            return None
    
        if pixels_per_mm_x == 0 or pixels_per_mm_y == 0:
            print("ERROR: pixels_per_mm cannot be zero!")
            return None
    
        try:
            # Convert focal length from mm to pixels
            fx = focal_length_mm * pixels_per_mm_x
            fy = focal_length_mm * pixels_per_mm_y
        
            # Principal point is already in pixels
            cx = principal_point_x_pixels
            cy = principal_point_y_pixels
        
            camera_matrix = np.array([
                [fx,  0,  cx],
                [0,  fy,  cy],
                [0,   0,   1]
            ], dtype=np.float64)
        
            return camera_matrix
        
        except Exception as e:
            print(f"ERROR in create_camera_matrix: {e}")
            return None