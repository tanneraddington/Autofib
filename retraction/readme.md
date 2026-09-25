# Description
ROS2 package for automated tissue retraction using deep learning models. This package provides multiple nodes for predicting push actions to retract tissue during surgical procedures, using both 2D image-based and 3D point cloud-based approaches. The package includes models for goal shape prediction and adaptive push action selection based on surgical context.

# Authors
* **Maintainer:** Tanner Watts (tanner.a.watts@vanderbilt.edu)

# API

## Nodes
* `retraction push2D_node`: 2D image-based push action prediction using CVAE models
  * Subscribes to topics:
    * `/ves_camera/image`, `sensor_msgs/Image`: Camera image from VES endoscope
    * `/trachea_pc`, `sensor_msgs/PointCloud2`: Trachea point cloud
    * `/bph_pc`, `sensor_msgs/PointCloud2`: Tumor/BPH point cloud
    * `/trajectory_pc`, `sensor_msgs/PointCloud2`: Cut trajectory point cloud (optional, for cut-conditioned model)
  * Publishes to topics:
    * `/start_pt`, `sensor_msgs/PointCloud2`: Predicted push start point
    * `/end_pt`, `sensor_msgs/PointCloud2`: Predicted push end point
  * Services and actions provided (server):
    * `push_action`, `aliss_ros_msg/srv/ActionRequest`: Service to request push action predictions
  * Command line arguments:
    * `--retract_count`: Initial retraction counter value (default: 0)
    * `--z_input`: Manual Z coordinate input (default: 0.0)
    * `--raw`: Use predicted Z value from model instead of tumor PC minimum (default: False)
    * `--cut_conditioned`: Use cut-conditioned model variant (default: False)

* `retraction pushVIBES_node`: 3D point cloud-based push action prediction using PushVIBES model
  * Subscribes to topics:
    * `/bph_pc`, `sensor_msgs/PointCloud2`: Current tumor/BPH point cloud
    * `/goal_pc`, `sensor_msgs/PointCloud2`: Goal point cloud for retraction
  * Publishes to topics:
    * `/start_pt`, `sensor_msgs/PointCloud2`: Predicted push start point
    * `/end_pt`, `sensor_msgs/PointCloud2`: Predicted push end point
    * `/goal_pc`, `sensor_msgs/PointCloud2`: Generated or published goal point cloud
    * `/generate_goal_flag`, `std_msgs/Bool`: Flag to trigger goal generation
  * Services and actions provided (server):
    * `push_action`, `aliss_ros_msg/srv/ActionRequest`: Service to request push action predictions

* `retraction adptive_push2d`: Adaptive 2D push action prediction that switches between models based on cut number
  * Subscribes to topics:
    * `/ves_camera/image`, `sensor_msgs/Image`: Camera image from VES endoscope
    * `/bph_pc`, `sensor_msgs/PointCloud2`: Tumor/BPH point cloud
    * `/trachea_pc`, `sensor_msgs/PointCloud2`: Trachea point cloud
    * `/cut_number`, `std_msgs/Int8`: Current cut number
    * `/first_median_cut_num`, `std_msgs/Int32`: Cut number threshold for model switching
  * Publishes to topics:
    * `/start_pt`, `sensor_msgs/PointCloud2`: Predicted push start point
    * `/end_pt`, `sensor_msgs/PointCloud2`: Predicted push end point
  * Services and actions provided (server):
    * `push_action`, `aliss_ros_msg/srv/ActionRequest`: Service to request push action predictions

* `retraction goal_service_node`: Generates goal point clouds for retraction planning
  * Subscribes to topics:
    * `/trachea_pc`, `sensor_msgs/PointCloud2`: Trachea point cloud
    * `/tumor_pc`, `sensor_msgs/PointCloud2`: Tumor point cloud
    * `/generate_goal_flag`, `std_msgs/Bool`: Flag to trigger goal generation
  * Publishes to topics:
    * `/goal_pc`, `sensor_msgs/PointCloud2`: Generated goal point cloud
  * Services and actions provided (server):
    * `goal_request`, `aliss_ros_msg/srv/GoalRequest`: Service to request goal point cloud generation

* `retraction data_collection_node`: Collects training data including images and robot poses
  * Subscribes to topics:
    * `/ves_camera/image`, `sensor_msgs/Image`: Camera image from VES endoscope
    * `/ves/left/joint/measured_cp`, `geometry_msgs/PoseStamped`: Left robot pose
    * `/ves/right/joint/measured_cp`, `geometry_msgs/PoseStamped`: Right robot pose
  * Publishes to topics:
    * None
  * Keyboard controls:
    * `S`: Save start point
    * `I`: Save image snapshot
    * `E`: Save end point
    * `C`: Save cut condition point
    * `Q`: Quit

## API changes

* No API changes documented yet.

# Installation

## Hardware dependencies
* GPU (CUDA-compatible) - Required for deep learning model inference (PyTorch)
* VES (Versatile Endoscopic System) - Required for real-time operation
* VES endoscope camera - Required for image-based models 

## OS requirements
* ROS2 Jazzy (or compatible ROS2 distribution)
* Python 3.8 or higher
* CUDA-capable GPU with appropriate drivers

### Python virtual environment setup:
  ```bash
  source /aliss_venv/bin/activate
  pip install -r requirements.txt
  ```
  

### Ubuntu/Debian packages:
  ```bash
  sudo apt install python3-pip python3-venv python3-dev
  sudo apt install ros-jazzy-rclpy ros-jazzy-sensor-msgs ros-jazzy-geometry-msgs ros-jazzy-cv-bridge
  ```


## ALISS dependencies
* `aliss_ros_msg` - ROS2 message and service definitions for ALISS system
  * Repository: https://github.com/ALISS-ARPAH/aliss_code
  * Required for `ActionRequest` and `GoalRequest` service definitions
* `aliss_state_machine aliss_state_machine` - Run aliss state machine node "retract" to run the retraction service
* `aliss_robot run_cao` - aliss_robot run_cao commands the start and end poses predicted by retraction
## Compilation
Full instructions, should be usable as copy/paste:
```bash
 cd aliss_core
 source aliss_venv/bin/activate
 ros2 run retraction push2d_node [Flags]
```


### Running individual nodes:

#### Push2D Node (2D image-based):
```bash
   ros2 run retraction push2D_node [--retract_count 0] [--z_input 0.0] [--raw False] [--cut_conditioned False]
```

#### PushVIBES Node (3D point cloud-based):
```bash
   ros2 run retraction pushVIBES_node
```

#### Adaptive Push2D Node (For BPH Surgery):
```bash
   ros2 run retraction adptive_push2d
```

#### Goal Service Node:
```bash
   ros2 run retraction goal_service_node
```

#### Data Collection Node:
```bash
   ros2 run retraction data_collection_node
```

## Expected results

When running the retraction system:

1. **Push2D Node**: 
   - Subscribes to camera images and optionally point clouds
   - Provides push action predictions via the `push_action` service
   - Publishes start and end points for visualization

2. **PushVIBES Node**:
   - Processes 3D point clouds to predict push actions
   - Generates goal point clouds if needed
   - Provides convergence detection when retraction is complete

3. **Goal Service Node**:
   - Generates goal point clouds based on the current tissue state (eventually to be replaced by DiffDef)
   - Can use deep learning models (DiffDef) or simple geometric transformations

4. **Data Collection Node**:
   - Saves training data to `data/bph_live/parsed_pickles_cc/`
   - Collects synchronized image and pose data for model training

The system is designed to work with the ALISS surgical robotics platform and VES endoscope system for automated tissue retraction during surgical procedures.



