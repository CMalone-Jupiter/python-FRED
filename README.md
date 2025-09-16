# python-FRED  
<!--  ![Zoe 2 img](assets/Zoe2-FRED.svg)  -->
<p align="center">
  <img src="assets/Zoe2-FRED.svg" alt="Zoe 2 img">
</p>  
This repository provides the devkit tools for working with the Flooded Road Environments Dataset (FRED). This autonomous vehicle dataset has been developed to enable research into the detection of flooded roads during on-road deployment. The dataset was collected using a Renault Zoe with custom modifications to enable autonomy, including front and rear Blackfly cameras, an Ouster OS1 LiDAR, and a GNSS-corrected IMU. Data has been collected using the vehicle's sensor stack from 5 separate locations around Brisbane, Australia, both during and after flooding events. Semantic labels are provided for images to enable the development of detection methods, and corresponding position information from the GNSS-corrected IMU has been provided across sequences to additionally enable localization research for these scenarios.

## Dataset Structure  
We adopt the following structure for FRED to include a KITTI-style format for the dataset and the native recording format using RTmaps.  
```
  ├── flooded                             # Location sequences captured during flooding events
  │   ├── KITTI-style                     # Sequences in a KITTI-style format
  │   |   ├── Cambogan_20250811_113017    # Sequence by location
  │   |   |   ├── back-imgs
  │   |   |   |   └── <timestamp>.png     # Images in 'png' format
  │   |   |   ├── front-imgs
  │   |   |   |   └── <timestamp>.png
  │   |   |   ├── imu
  │   |   |   |   └── <timestamp>.txt     # IMU data formatted as a 'txt' file
  │   |   |   ├── ouster
  │   |   |   |   └── <timestamp>.bin     # Point clouds formatted as a binary file
  │   |   |   └── utm
  │   |   |       └── <timestamp>.txt     # UTM locations formatted as a 'txt' file
  │   |   ├── ...
  │   |   └── ...
  │   └── native-RTmaps                   # Sequences in native recording format
  │       ├── Cambogan_20250811_113017    # Sequence by location
  │       |   ├── Camera_Rec              # Recording files for image playback
  │       |   ├── IMU_Info_Rec            # Recording files for IMU playback
  │       |   └── Ouster_Rec              # Recording files for LiDAR playback
  │       ├── ...
  │       └── ...
  │
  └── dry                             # Location sequences captured while 'dry'
      ├── KITTI-style                     
      └── native-RTmaps              
```  
