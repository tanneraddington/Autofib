# Uterine Project
For the MEDLab R01 project

---
# Computer Setup

## Libraries
#### Eigen
```console
$ sudo apt install libeigen3-dev
```
#### Boost

```console
$ sudo apt install libboost-all-dev
```
### Dynamixel
```console
$ sudo apt-get install ros-[ROS Distribution]-dynamixel-sdk
```

## CMakeList
For each library that's included in a package, you need to add the following to your CMakeList.txt
```CMake
find_package(PACKAGE_NAME REQUIRED)

ament_target_dependencies(EXECUTABLE_NAME
  PACKAGE_NAME
)
```

Here's an example for the motor_node
```CMake

find_package(dynamixel_sdk REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

ament_target_dependencies(motor_node
  dynamixel_sdk
  rclcpp
  std_msgs
)
```

# Notes on Install

## Ubuntu Install
Follow instructions here: https://ubuntu.com/tutorials/install-ubuntu-desktop#11-dont-forget-to-update

## ROS2 Install
Follow instructions here: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html 

# Running the System

## Homing
Homing the robot is done when no other control signal is being sent to the motors. Motors automatically home upon start-up. Motor node must be running to home motors. Send ```$ros2 topic pub -1 /set_home std_msgs/msgs/Int32 "{data: 1}"``` to send the motors to the nearest home. This will align all of the motor couplers so you can insert or remove cartridges.

## Running everything together
1. Make sure motors are on and connected and controllers are connected and powered.
2. Start the motor node ``` $ros2 launch motor_pkg motor_pkg_launch ``` to run without comments or ```$ros2 run motor_pkg motor_node --ros-args -p comments:=true```. Motors will home upon start-up.
3. Tweak the home position to align the tubes with the home position. ```$ros2 run keyboard_direct keyboard_direct ``` will start the manual keyboard jogging. You can then use keystrokes to jog each joint to align the home positions. Key controls in table below.
4. Start the controllers in a new terminal window by running the command ```$ros2 launch virtuoso_package virtuoso_pkg_launch.xml```
5. The motors are now controlled by the controllers. To stop control, simply ctrl+C in the node you ran the launch file in.
6. To home the motors, make sure the controller node is not running, then run this command: ```$ros2 topic pub -1 /set_home std_msgs/msgs/Int32 "{data: 1}"```. You can now remove the cartridges or re-connect the controllers.

Note: Camera encoder position, right controller down angle, and left controller down angle will be saved to csv file in /ros2_ws each time MotorNode, virtuoso_ui_node_l, and virtuoso_ui_node_r are launched. 


