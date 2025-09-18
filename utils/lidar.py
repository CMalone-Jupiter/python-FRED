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
        Tr4: 4x4 homogeneous transform from velodyne -> camera.
        Returns pts_cam (N,3) in camera coordinates.
        """
        n = self.points.shape[0]
        pts_h = np.hstack((self.points, np.ones((n,1))))        # (N,4)
        pts_cam_h = (self.Tr4 @ pts_h.T).T                       # (N,4)
        return pts_cam_h[:, :3]

    def project_to_image(self,pts_cam, P2, R0_rect=None):
        """
        pts_cam: (N,3) in camera coordinates (before rectification)
        R0_rect: 3x3 rectification matrix (if None, identity)
        P2: 3x4 projection matrix
        Returns:
        pts_uv: (N,2) pixel coordinates (float)
        depth: (N,) Z in camera rectified frame
        valid: boolean mask where depth > 0
        """
        if R0_rect is None:
            R0_rect = np.eye(3)
        # apply rectification
        pts_rect = (R0_rect @ pts_cam.T).T      # (N,3)
        n = pts_rect.shape[0]
        pts_rect_h = np.hstack((pts_rect, np.ones((n,1))))  # (N,4)
        proj = (P2 @ pts_rect_h.T).T            # (N,3)
        z = proj[:, 2].copy()
        valid = z > 1e-6
        uv = np.zeros((n,2), dtype=float)
        uv[valid,0] = proj[valid,0] / z[valid]
        uv[valid,1] = proj[valid,1] / z[valid]
        return uv, z, valid