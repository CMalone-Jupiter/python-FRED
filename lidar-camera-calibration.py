import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
import argparse
import os
 
class LiDARCameraCalibrator:
    def __init__(self, point_cloud_path, image_path, camera_matrix, dist_coeffs):
        """
        Initialize the calibrator with point cloud and image data
       
        Args:
            point_cloud_path (str): Path to LiDAR point cloud file (.pcd, .ply, .bin, etc.)
            image_path (str): Path to camera image
            camera_matrix (np.array): 3x3 camera intrinsic matrix
            dist_coeffs (np.array): Camera distortion coefficients
        """
        # Load point cloud
        self.points_3d = self.load_point_cloud(point_cloud_path)
       
        # Load image
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
       
        # Camera parameters
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
 
        # self.fixed_translation = np.array([0.676, 0.0, 1.486])
        # self.fixed_transform = np.eye(4)
        # self.fixed_transform[:3, 3] = self.fixed_translation
       
        # Initial transformation parameters (adjustable via trackbars)
        # tx=1.932+0.676, ty=0.25+0, tz=1.4+1.486, rx=0, ry=-2.0, rz=1.5
        self.tx = -1.16  # Translation X
        self.ty = -0.23  # Translation Y  
        self.tz = 0.42  # Translation Z
        self.rx = 0.0  # Rotation X (degrees)
        self.ry = 2.0  # Rotation Y (degrees)
        self.rz = -0.55  # Rotation Z (degrees)
       
        # Trackbar ranges (scaled by 1000 for precision)
        self.t_range = 50000  # ±50m in mm
        self.r_range = 18000  # ±180 degrees in 0.01 degree units
       
        # Setup OpenCV window and trackbars
        self.setup_gui()
   
    def load_point_cloud(self, file_path):
        """
        Load point cloud from various formats including .bin files
       
        Args:
            file_path (str): Path to point cloud file
           
        Returns:
            np.array: Nx3 array of 3D points
        """
        file_ext = os.path.splitext(file_path)[1].lower()
       
        if file_ext == '.bin':
            # Load binary point cloud (common format for KITTI, nuScenes, etc.)
            # Assumes format: [x, y, z, intensity] or [x, y, z, intensity, ring, time, etc.]
            points = np.fromfile(file_path, dtype=np.float32)
           
            # Determine the number of fields per point
            # Common formats: 4 (x,y,z,intensity), 5 (x,y,z,intensity,ring), 6+ (additional fields)
            if len(points) % 4 == 0:
                points = points.reshape(-1, 4)
                points_3d = points[:, :3]  # Take only x, y, z
                self.intensities = points[:, 3]  # Store intensity for potential use
                # print(self.intensities.max())
                # print(self.intensities.min())
                print(f"Loaded {len(points_3d)} points from .bin file (4 fields per point)")
            elif len(points) % 6 == 0:
                points = points.reshape(-1, 6)
                points_3d = points[:, :3]  # Take only x, y, z
                self.intensities = points[:, 3]  # Store intensity
                print(f"Loaded {len(points_3d)} points from .bin file (6 fields per point)")
            else:
                # Try to infer the structure or default to 3D points only
                # Assume the most common case where points are stored as [x,y,z,...]
                n_fields = 4  # Default assumption
                if len(points) % 3 == 0 and len(points) % 4 != 0:
                    n_fields = 3
               
                points = points.reshape(-1, n_fields)
                points_3d = points[:, :3]
                if n_fields > 3:
                    self.intensities = points[:, 3]
                else:
                    self.intensities = None
                print(f"Loaded {len(points_3d)} points from .bin file ({n_fields} fields per point)")
           
            return points_3d
           
        else:
            # Use Open3D for standard formats (.pcd, .ply, etc.)
            try:
                pcd = o3d.io.read_point_cloud(file_path)
                points_3d = np.asarray(pcd.points)
                self.intensities = None  # Open3D formats don't always have intensity
                print(f"Loaded {len(points_3d)} points using Open3D")
                return points_3d
            except Exception as e:
                raise ValueError(f"Could not load point cloud {file_path}: {str(e)}")
       
       
    def setup_gui(self):
        """Setup OpenCV window and trackbars for real-time adjustment"""
        cv2.namedWindow('LiDAR-Camera Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('LiDAR-Camera Calibration', 1200, 800)
       
        # Create trackbars for translation (in mm for precision)
        cv2.createTrackbar('TX (mm)', 'LiDAR-Camera Calibration',
                          int(self.tx * 1000) + self.t_range, 2 * self.t_range, self.update_transform)
        cv2.createTrackbar('TY (mm)', 'LiDAR-Camera Calibration',
                          int(self.ty * 1000) + self.t_range, 2 * self.t_range, self.update_transform)
        cv2.createTrackbar('TZ (mm)', 'LiDAR-Camera Calibration',
                          int(self.tz * 1000) + self.t_range, 2 * self.t_range, self.update_transform)
       
        # Create trackbars for rotation (in 0.01 degree units)
        cv2.createTrackbar('RX (0.01°)', 'LiDAR-Camera Calibration',
                          int(self.rx * 100) + self.r_range, 2 * self.r_range, self.update_transform)
        cv2.createTrackbar('RY (0.01°)', 'LiDAR-Camera Calibration',
                          int(self.ry * 100) + self.r_range, 2 * self.r_range, self.update_transform)
        cv2.createTrackbar('RZ (0.01°)', 'LiDAR-Camera Calibration',
                          int(self.rz * 100) + self.r_range, 2 * self.r_range, self.update_transform)
       
        # Additional controls
        cv2.createTrackbar('Point Size', 'LiDAR-Camera Calibration', 2, 10, self.update_transform)
        cv2.createTrackbar('Max Distance (m)', 'LiDAR-Camera Calibration', 100, 200, self.update_transform)
        cv2.createTrackbar('Min Distance (m)', 'LiDAR-Camera Calibration', 0, 50, self.update_transform)
        cv2.createTrackbar('Intensity Filter', 'LiDAR-Camera Calibration', 0, 100, self.update_transform)
   
    def set_initial_values(self, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
        """Set initial transformation values"""
        self.tx, self.ty, self.tz = tx, ty, tz
        self.rx, self.ry, self.rz = rx, ry, rz
       
        # Update trackbars
        cv2.setTrackbarPos('TX (mm)', 'LiDAR-Camera Calibration', int(tx * 1000) + self.t_range)
        cv2.setTrackbarPos('TY (mm)', 'LiDAR-Camera Calibration', int(ty * 1000) + self.t_range)
        cv2.setTrackbarPos('TZ (mm)', 'LiDAR-Camera Calibration', int(tz * 1000) + self.t_range)
        cv2.setTrackbarPos('RX (0.01°)', 'LiDAR-Camera Calibration', int(rx * 100) + self.r_range)
        cv2.setTrackbarPos('RY (0.01°)', 'LiDAR-Camera Calibration', int(ry * 100) + self.r_range)
        cv2.setTrackbarPos('RZ (0.01°)', 'LiDAR-Camera Calibration', int(rz * 100) + self.r_range)
   
    def update_transform(self, val):
        """Callback function for trackbar updates"""
        # Get current trackbar values
        self.tx = (cv2.getTrackbarPos('TX (mm)', 'LiDAR-Camera Calibration') - self.t_range) / 1000.0
        self.ty = (cv2.getTrackbarPos('TY (mm)', 'LiDAR-Camera Calibration') - self.t_range) / 1000.0
        self.tz = (cv2.getTrackbarPos('TZ (mm)', 'LiDAR-Camera Calibration') - self.t_range) / 1000.0
        self.rx = (cv2.getTrackbarPos('RX (0.01°)', 'LiDAR-Camera Calibration') - self.r_range) / 100.0
        self.ry = (cv2.getTrackbarPos('RY (0.01°)', 'LiDAR-Camera Calibration') - self.r_range) / 100.0
        self.rz = (cv2.getTrackbarPos('RZ (0.01°)', 'LiDAR-Camera Calibration') - self.r_range) / 100.0
       
        # Update visualization
        self.visualize_projection()
   
    def get_transformation_matrix(self):
        """Calculate 4x4 transformation matrix from current parameters"""
        # Create rotation matrix from Euler angles (XYZ order)
        rotation = Rotation.from_euler('xyz', [self.rx, self.ry, self.rz], degrees=True)
        # R = rotation.as_matrix()
 
        coord_transform = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
 
        R = rotation.as_matrix() #@ coord_transform
       
        # Create translation vector
        t = np.array([self.tx, self.ty, self.tz])
       
        # Build 4x4 transformation matrix
        variable_T = np.eye(4)
        variable_T[:3, :3] = R
        variable_T[:3, 3] = t
 
        # T = variable_T @ self.fixed_transform
        T = variable_T
 
        # Convert combined transform to camera coordinate frame
        C4 = np.eye(4)
        C4[:3, :3] = coord_transform   # maps robot -> camera
        T_camera = C4 @ T
       
        return T_camera
   
    def project_points(self):
        """Project 3D LiDAR points to image coordinates"""
        if len(self.points_3d) == 0:
            return np.array([]), np.array([])
       
        # Apply transformation
        T = self.get_transformation_matrix()
        points_homo = np.column_stack([self.points_3d, np.ones(len(self.points_3d))])
        transformed_points = (T @ points_homo.T).T[:, :3]
       
        # Filter points that are in front of camera
        valid_mask = transformed_points[:, 2] > 0
        valid_points = transformed_points[valid_mask]
        valid_inten = self.intensities[valid_mask]
       
        if len(valid_points) == 0:
            return np.array([]), np.array([])
       
        # Filter by distance
        max_dist = cv2.getTrackbarPos('Max Distance (m)', 'LiDAR-Camera Calibration')
        min_dist = cv2.getTrackbarPos('Min Distance (m)', 'LiDAR-Camera Calibration')
        distances = np.linalg.norm(valid_points, axis=1)
        dist_mask = (distances >= min_dist) & (distances <= max_dist)
        valid_points = valid_points[dist_mask]
        valid_inten = valid_inten[dist_mask]
        valid_distances = distances[dist_mask]
       
        # Filter by intensity if available
        if hasattr(self, 'intensities') and self.intensities is not None:
            intensity_threshold = cv2.getTrackbarPos('Intensity Filter', 'LiDAR-Camera Calibration') / 100.0
            # Apply original valid_mask and dist_mask to intensities
            original_valid_mask = np.arange(len(self.points_3d))[valid_mask[:len(valid_mask)]][:len(valid_points)]
            if len(original_valid_mask) > 0 and len(original_valid_mask) <= len(self.intensities):
                valid_intensities = self.intensities[original_valid_mask]
                # Normalize intensities to 0-1 range
                if len(valid_intensities) > 0:
                    intensity_norm = (valid_intensities - np.min(valid_intensities)) / (np.max(valid_intensities) - np.min(valid_intensities) + 1e-8)
                    intensity_mask = intensity_norm >= intensity_threshold
                    valid_points = valid_points[intensity_mask]
                    valid_inten = valid_inten[intensity_mask]
                    valid_distances = valid_distances[intensity_mask]
       
        if len(valid_points) == 0:
            return np.array([]), np.array([])
       
        # Project to image coordinates
        rvec = np.zeros(3)  # No additional rotation
        tvec = np.zeros(3)  # No additional translation
       
        image_points, _ = cv2.projectPoints(
            valid_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
       
        image_points = image_points.reshape(-1, 2)
       
        # Filter points within image bounds
        h, w = self.height, self.width
        valid_img_mask = ((image_points[:, 0] >= 0) & (image_points[:, 0] < w) &
                         (image_points[:, 1] >= 0) & (image_points[:, 1] < h))
       
        return image_points[valid_img_mask], valid_distances[valid_img_mask], valid_inten[valid_img_mask]
   
    def visualize_projection(self):
        """Update the visualization with current transformation"""
        # Reset image
        self.image = self.original_image.copy()
       
        # Project points
        projected_points, distances, intensities = self.project_points()
       
        if len(projected_points) == 0:
            cv2.putText(self.image, "No valid points to display", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Color points by distance (closer = red, farther = blue)
            max_dist = cv2.getTrackbarPos('Max Distance (m)', 'LiDAR-Camera Calibration')
            normalized_dist = distances.clip(0, 100) / min(max_dist,distances.max())
            normalized_intensity = intensities/255
           
            point_size = cv2.getTrackbarPos('Point Size', 'LiDAR-Camera Calibration')
            if point_size == 0:
                point_size = 1
               
            # for i, (point, dist) in enumerate(zip(projected_points, normalized_dist)):
            for i, (point, dist) in enumerate(zip(projected_points, normalized_intensity)):
                # Color from red (close) to blue (far)
                color = (255 * (1 - dist), 0, 255 * dist)
                cv2.circle(self.image, (int(point[0]), int(point[1])),
                          point_size, color, -1)
       
        # Add parameter information
        info_text = [
            f"TX: {self.tx:.3f}m  TY: {self.ty:.3f}m  TZ: {self.tz:.3f}m",
            f"RX: {self.rx:.2f}°  RY: {self.ry:.2f}°  RZ: {self.rz:.2f}°",
            f"Projected points: {len(projected_points)}",
            "Press 'q' to quit, 's' to save parameters, 'r' to reset"
        ]
       
        for i, text in enumerate(info_text):
            cv2.putText(self.image, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(self.image, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
       
        cv2.imshow('LiDAR-Camera Calibration', self.image)
   
    def save_parameters(self, filename="calibration_params.txt"):
        """Save current transformation parameters to file"""
        T = self.get_transformation_matrix()
       
        with open(filename, 'w') as f:
            f.write("LiDAR-Camera Calibration Parameters\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Translation (meters):\n")
            f.write(f"  TX: {self.tx:.6f}\n")
            f.write(f"  TY: {self.ty:.6f}\n")
            f.write(f"  TZ: {self.tz:.6f}\n\n")
            f.write(f"Rotation (degrees):\n")
            f.write(f"  RX: {self.rx:.6f}\n")
            f.write(f"  RY: {self.ry:.6f}\n")
            f.write(f"  RZ: {self.rz:.6f}\n\n")
            f.write("4x4 Transformation Matrix:\n")
            f.write(str(T) + "\n")
       
        print(f"Parameters saved to {filename}")
   
    def run(self):
        """Start the interactive calibration tool"""
        print("LiDAR-Camera Calibration Tool")
        print("Use trackbars to adjust transformation parameters")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current parameters")
        print("  'r' - Reset to initial values")
       
        # Initial visualization
        self.visualize_projection()
       
        while True:
            key = cv2.waitKey(30) & 0xFF
           
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_parameters()
            elif key == ord('r'):
                self.set_initial_values()
                self.visualize_projection()
       
        cv2.destroyAllWindows()
 
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
    # Debug: Check inputs
    # print(f"Input values:")
    # print(f"  focal_length_mm: {focal_length_mm}")
    # print(f"  principal_point_x_pixels: {principal_point_x_pixels}")
    # print(f"  principal_point_y_pixels: {principal_point_y_pixels}")
    # print(f"  pixels_per_mm_x: {pixels_per_mm_x}")
    # print(f"  pixels_per_mm_y: {pixels_per_mm_y}")
   
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
       
        # print(f"DEBUG: Camera matrix calculation:")
        # print(f"  fx = {focal_length_mm} * {pixels_per_mm_x} = {fx}")
        # print(f"  fy = {focal_length_mm} * {pixels_per_mm_y} = {fy}")
        # print(f"  cx = {cx}")
        # print(f"  cy = {cy}")
       
        camera_matrix = np.array([
            [fx,  0,  cx],
            [0,  fy,  cy],
            [0,   0,   1]
        ], dtype=np.float64)
       
        # print(f"DEBUG: Final camera matrix:\n{camera_matrix}")
       
        return camera_matrix
       
    except Exception as e:
        print(f"ERROR in create_camera_matrix: {e}")
        return None
 
def create_sample_data():
    """Create sample point cloud and camera parameters for testing"""
    # Generate sample point cloud (random points in a cube)
    np.random.seed(42)
    n_points = 1000
    points = np.random.uniform(-10, 10, (n_points, 3))
    points[:, 2] += 20  # Move points in front of camera
   
    # Add intensity values
    intensities = np.random.uniform(0, 255, n_points)
   
    # Create .bin file (KITTI format: x, y, z, intensity)
    bin_data = np.column_stack([points, intensities]).astype(np.float32)
    bin_data.tofile("sample_pointcloud.bin")
   
    # Also create .pcd for compatibility
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("sample_pointcloud.pcd", pcd)
   
    # Create sample camera matrix (typical values)
 
    focal_length_mm = 12.0            # Your focal length in mm
    principal_point_x_pixels = 960    # Principal point X in pixels
    principal_point_y_pixels = 600    # Principal point Y in pixels  
    pixels_per_mm_x = 170.648           # Your pixels per mm in X
    pixels_per_mm_y = 170.648           # Your pixels per mm in Y
   
    # Create camera matrix
    camera_matrix = create_camera_matrix(
        focal_length_mm, principal_point_x_pixels, principal_point_y_pixels,
        pixels_per_mm_x, pixels_per_mm_y
    )
    # camera_matrix = np.array([
    #     [800, 0, 320],
    #     [0, 800, 240],
    #     [0, 0, 1]
    # ], dtype=np.float32)
   
    # Sample distortion coefficients
    # dist_coeffs = np.array([0.1, -0.2, 0, 0, 0.1], dtype=np.float32)
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
   
    # Create a simple test image
    test_image = np.zeros((1200, 1920, 3), dtype=np.uint8)
    cv2.putText(test_image, "Sample Camera Image", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite("sample_image.jpg", test_image)
   
    return camera_matrix, dist_coeffs
 
 
def main():
    parser = argparse.ArgumentParser(description='LiDAR-Camera Calibration Tool')
    parser.add_argument('--pcd', type=str, help='Path to point cloud file (.pcd, .ply, .bin, etc.)')
    parser.add_argument('--image', type=str, help='Path to camera image')
    parser.add_argument('--demo', action='store_true', help='Run with sample data')
   
    args = parser.parse_args()
   
    if args.demo or (not args.pcd or not args.image):
        print("Creating sample data for demonstration...")
        camera_matrix, dist_coeffs = create_sample_data()
        pcd_path = "sample_pointcloud.bin"  # Use .bin file for demo
        img_path = "sample_image.jpg"
    else:
        pcd_path = args.pcd
        img_path = args.image
       
        # You'll need to provide your camera calibration parameters here
        focal_length_mm = 12.0            # Your focal length in mm
        principal_point_x_pixels = 960    # Principal point X in pixels
        principal_point_y_pixels = 600    # Principal point Y in pixels  
        pixels_per_mm_x = 170.648           # Your pixels per mm in X
        pixels_per_mm_y = 170.648           # Your pixels per mm in Y
       
        # Create camera matrix
        camera_matrix = create_camera_matrix(
            focal_length_mm, principal_point_x_pixels, principal_point_y_pixels,
            pixels_per_mm_x, pixels_per_mm_y
        )
        # camera_matrix = np.array([
        #     [800, 0, 320],
        #     [0, 800, 240],
        #     [0, 0, 1]
        # ], dtype=np.float32)
       
        # Sample distortion coefficients
        # dist_coeffs = np.array([0.1, -0.2, 0, 0, 0.1], dtype=np.float32)
        dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
   
    # Check if files exist
    if not os.path.exists(pcd_path):
        print(f"Error: Point cloud file {pcd_path} not found")
        return
   
    if not os.path.exists(img_path):
        print(f"Error: Image file {img_path} not found")
        return
   
    # Initialize calibrator
    calibrator = LiDARCameraCalibrator(pcd_path, img_path, camera_matrix, dist_coeffs)
   
    # Set initial transformation values (modify these as needed)
    # calibrator.set_initial_values(tx=1.932, ty=0, tz=1.4, rx=0, ry=-2.0, rz=1.5)
    # calibrator.set_initial_values(tx=1.932, ty=0.25, tz=1.4, rx=0.0, ry=-2.0, rz=1.5)
    calibrator.set_initial_values(tx=-1.256, ty=-0.1, tz=0.5, rx=0.0, ry=2.2, rz=-0.55)
    # calibrator.set_initial_values(tx=-1.16, ty=-0.23, tz=0.42, rx=0.0, ry=2.0, rz=-0.55)
    # calibrator.set_initial_values(tx=1.932, ty=0.25, tz=1.486, rx=0, ry=-2.0, rz=1.5)
    # calibrator.set_initial_values(tx=0.23, ty=-0.42, tz=-1.16, rx=-2.5, ry=0.46, rz=-0.7)
    # calibrator.set_initial_values(tx=0.192, ty=-0.641, tz=-1.474, rx=-2.51, ry=0.46, rz=-0.69)
   
    # Run the calibration tool
    calibrator.run()
 
 
if __name__ == "__main__":
    main(