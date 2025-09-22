import numpy as np
import os
from utils.utils import read_calib_file

class PointCloud:
    def __init__(self, file_path, calib_path):

        self.points = self.load_pointcloud(file_path)
        calibration = read_calib_file(calib_path)
        self.P2, self.R0, self.Tr4 = self.get_matrices(calibration)


    
    def load_pointcloud(self, file_path):
        """
        Load point cloud from various formats including .bin files
        Assumes each point has 4 values (x, y, z, reflectivity)
        
        Args:
            file_path (str): Path to point cloud file
            
        Returns:
            np.array: Nx4 array of 3D points with 
        """

        file_ext = os.path.splitext(file_path)[1].lower()
            
        if not file_ext == '.bin':
            raise ValueError(f"Given file is not binary format: {file_path}")
        
        points = np.fromfile(file_path, dtype=np.float32)

        if len(points) % 4 == 0:
            points_array = points.reshape(-1, 4)

        return points_array
    

    def get_matrices(self,calib):
        # P2 (3x4)
        P2 = calib['P2'].reshape(3,4)
        # R0_rect (3x3) or identity
        R0 = calib.get('R0_rect', np.array([1,0,0,0,1,0,0,0,1])).reshape(3,3)
        # Tr_ouster_to_cam (3x4)
        Tr = calib['Tr_ouster_to_cam'].reshape(3,4)
        # Make 4x4 homogeneous
        Tr4 = np.vstack((Tr, [0,0,0,1]))

        return P2, R0, Tr4

    def points_ouster_to_cam(self,):
        """
        pts_ouster: (N,3) numpy array in LiDAR frame.
        Tr4: 4x4 homogeneous transform from ouster -> camera.
        Returns pts_cam (N,3) in camera coordinates.
        """
        n = self.points.shape[0]
        pts_h = np.column_stack((self.points[:,:3], np.ones((n))))        # (N,4)
        pts_cam_h = (self.Tr4 @ pts_h.T).T[:,:3]                       # (N,4)

        valid = pts_cam_h[:,2] > 1e-6

        distances = np.linalg.norm(pts_cam_h[valid,:], axis=1)        

        return pts_cam_h[valid,:], distances, self.points[valid,3]
