import os
import numpy as np
import cv2
from PIL import Image
from utils.utils import read_calib_file

class ImageData():
    def __init__(self, file_path, calib_path, label_path=None):
        self.image = cv2.imread(file_path)

        intrinsics = read_calib_file(calib_path)
        focal_len = intrinsics['focal_len'][0]
        principal_x = intrinsics['principal_x'][0]
        principal_y = intrinsics['principal_y'][0]
        pp_mm_x = intrinsics['pp_mm_x'][0]
        pp_mm_y = intrinsics['pp_mm_y'][0]

        # print(f"{focal_len}, {principal_x}, {principal_y}, {pp_mm_x}, {pp_mm_y}")

        self.camera_matrix = self.create_camera_matrix(focal_len, principal_x, principal_y, pp_mm_x, pp_mm_y)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self.semantic_classes = None
        self.colour_label = None
        self.semantic_classes = {'road': [0, 0, 128], 'water': [0, 128, 0], 'other': [0, 0, 0]} # In the opencv BGR convention

        if label_path is not None:        
            if os.path.exists(label_path):
                label_img = cv2.imread(label_path)
            else:
                # Catch for no label existing #
                print(f"Could not load label \'{label_path}\'. Using blank labels.")
                label_img = np.zeros(self.image.shape).astype(np.uint8)
            self.colour_label = cv2.resize(label_img, (self.image.shape[1], self.image.shape[0]))

    def create_camera_matrix(self, focal_length_mm, principal_point_x_pixels, principal_point_y_pixels,
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
        
    def project_points(self, points, colours, cmap):
        # Project to image coordinates
        rvec = np.zeros(3)  # No additional rotation
        tvec = np.zeros(3)  # No additional translation

        image_points, _ = cv2.projectPoints(points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

        image_points = image_points.reshape(-1, 2)

        # Filter points within image bounds
        h, w = self.image.shape[0], self.image.shape[1]
        valid_img_mask = ((image_points[:, 0] >= 0) & (image_points[:, 0] < w) &
                            (image_points[:, 1] >= 0) & (image_points[:, 1] < h))

        points2project = image_points[valid_img_mask]
        colour2project = colours[valid_img_mask]/255

        # print(points2project.shape)

        img_vis = self.image.copy()
        # Draw points
        for (point, c) in zip(points2project.astype(int), colour2project):
            r, g, b, _ = cmap(c)
            colour = (r*255, g*255, b*255)
            cv2.circle(img_vis, (int(point[0]), int(point[1])), 3, colour, -1)  # -1 = filled circle

        return img_vis