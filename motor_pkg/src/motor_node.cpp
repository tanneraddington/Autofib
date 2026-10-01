// Copyright 2021 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*******************************************************************************
// This example is written for DYNAMIXEL X(excluding XL-320) and MX(2.0) series with U2D2.
// For other series, please refer to the product eManual and modify the Control Table addresses and other definitions.
// To test this example, please follow the commands below.
//
// Open terminal #1
// $ ros2 run dynamixel_sdk_examples read_write_node
//
// Open terminal #2 (run one of below commands at a time)
// $ ros2 topic pub -1 /set_position dynamixel_sdk_custom_interfaces/SetPosition "{id: 1, position: 1000}"
// $ ros2 service call /get_position dynamixel_sdk_custom_interfaces/srv/GetPosition "id: 1"
//
// Author: Will Son

// Note: this example does not apply with our now edited code
*******************************************************************************/

// Takes in array giving position and rotation commands and sends motors to those positions

//Can change motor operating mode in first dxl_comm_result definition in setupDynamixel function

#include <cstdio>
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <array>
#include <cmath>

#include <dynamixel_sdk/dynamixel_sdk.h>  //Dynamixel library

#include <rclcpp/rclcpp.hpp>
#include "rcutils/cmdline_parser.h"

#include "motor_node.hpp"
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
using std::placeholders::_1;


// Control table address for X series (except XL-320)
#define ADDR_OPERATING_MODE 11
#define ADDR_TORQUE_ENABLE 64
#define ADDR_GOAL_POSITION 116
#define ADDR_GOAL_CURRENT 102
#define ADDR_PRESENT_POSITION 132

//  Protocol version
#define PROTOCOL_VERSION 2.0 // Default Protocol version of DYNAMIXEL X series.

// Default setting
#define BAUDRATE 115200             // Default Baudrate of DYNAMIXEL X series
//#define BAUDRATE 57600
#define DEVICE_NAME "/dev/ttyUSB0" // [Linux]: "/dev/ttyUSB*", [Windows]: "COM*"
#define ENCODER_TICKS_PER_REV 4096
#define ENCODER_TICKS_PER_REV_CAMERA 4096
#define LEAD_SCREW_REV_PER_M 1/.02 // 20mm/rotation
#define REV_PER_RADIAN 1/(2*M_PI)

//Define physical parameters for fwkin
#define CURVATURE_OUTER 107
#define CURVATURE_INNER 0
#define E 8300
#define OD_OUTER 0.00145
#define OD_INNER 0.0008
#define ID_OUTER 0.0011
#define ID_INNER 0.0005


dynamixel::PortHandler *portHandler;
dynamixel::PacketHandler *packetHandler;

//Initializing constants
uint8_t dxl_error = 0;
int dxl_comm_result;
uint8_t param_goal_position[4] = {0};
uint8_t param_goal_rotation[4] = {0};
bool at_rest = false;
bool first_camera_loop = true;
bool first_home_loop = true;
bool check = false;
float camera_s2 = (67.5/75) * 45 * M_PI/180;
float camera_s3 = (67.5/75) * 80 * M_PI/180;
float camera_scaler_forward = -75/.042 *M_PI/180; //rad/m; 42mm outer tube translation = 75deg rotation
float camera_scaler_backward = -70/.024 * M_PI/180; //needs 1:1 scale when returning home
double scalar = camera_scaler_forward;
double rot = 0;
double pos = 0;
float size = 0;
double camera_pos;
bool motor_exist = false;
float override;

dynamixel::GroupSyncWrite * groupSyncWritePosition;//(portHandler, packetHandler, ADDR_GOAL_POSITION, 4);
dynamixel::GroupSyncWrite * groupSyncWriteCurrent;//(portHandler, packetHandler, ADDR_GOAL_POSITION, 4);
dynamixel::GroupSyncRead * groupSyncRead;


//////////////////////////////////////////////////////////////////////////////////////
// smoother_uterus integration: per-arm joint state feedback
//
// Publishes /robot/<side>/joint/measured_jp (sensor_msgs/JointState) for the smoother and
// the visual_servoing action server. Positions are PHYSICAL joint values (command + keyboard
// offset), ordered [inner_rotation, outer_rotation, inner_translation, outer_translation],
// in rad / m, i.e. the same per-arm order as /robot/state/current_state.
//
// Params:
//   joint_readback      (bool,   default false) true: read present positions from the motors
//                                               false: publish the last command actually sent
//   joint_state_rate_hz (double, default 30)    republish/read rate so every camera frame has a
//                                               joint state close in time, even when idle
//   camera_tracking     (bool,   default true)  false: hold the camera motor at camera_fixed_angle
//   camera_fixed_angle  (double, default 0.0)   rad, used when camera_tracking is false
//
// Kept as file-scope state (like the rest of this file) so motor_node.hpp doesn't need to change.
//////////////////////////////////////////////////////////////////////////////////////
namespace {
  constexpr double TICKS_PER_M   = ENCODER_TICKS_PER_REV / 0.02;          // lead screw: 20 mm / rev
  constexpr double TICKS_PER_RAD = ENCODER_TICKS_PER_REV / (2.0 * M_PI);
  //constexpr double CAMERA_MAX_ANGLE = 1.291;                              // rad, same clamp as position_callback

  // motor_array layout: right [0]=ot [1]=or [2]=it [3]=ir, left [4]=ot [5]=or [6]=it [7]=ir
  // published order per arm:  [ir, or, it, ot]
  constexpr std::array<int, 4> RIGHT_ARM_IDX = {3, 1, 2, 0};
  constexpr std::array<int, 4> LEFT_ARM_IDX  = {7, 5, 6, 4};

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_right;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_left;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_camera;
  rclcpp::TimerBase::SharedPtr js_timer;

  std::array<double, 8> joint_phys{};  // physical joint values, motor_array layout
  double camera_angle_rad = 0.0;
  bool have_joint_phys = false;
  bool joint_readback = false;
  
  constexpr uint8_t CAMERA_ID = 10;
  constexpr float CAMERA_HOME_DEG = 15.0f;   // camera angle at the startup (home) position
  constexpr float CAMERA_MIN_DEG  = 17.0f;
  constexpr float CAMERA_MAX_DEG  = 88.0f;
  constexpr float CAMERA_TICKS_PER_DEG = ENCODER_TICKS_PER_REV / 360.0f;
  bool camera_ready = false;
  int32_t camera_home_ticks = 0;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr camera_sub;

  void publish_camera_state(const rclcpp::Time & stamp)
  {
    sensor_msgs::msg::JointState js;
    js.header.stamp = stamp;
    js.name = {"camera"};
    js.position = {camera_angle_rad};
    js_pub_camera->publish(js);
  }

  // CAMERA: call after openPort and after setupDynamixel(BROADCAST_ID), which leaves ID 10 in
  // mode 5 with torque on. Torque off -> Extended Position (4) -> read home -> torque on.
  bool setupCamera()
  {
    auto logger = rclcpp::get_logger("motor_node");

    if (packetHandler->ping(portHandler, CAMERA_ID, &dxl_error) != COMM_SUCCESS) {
      RCLCPP_WARN(logger, "Camera motor (ID %d) not detected", CAMERA_ID);
      return false;
    }
    if (packetHandler->write1ByteTxRx(portHandler, CAMERA_ID, ADDR_TORQUE_ENABLE, 0, &dxl_error) != COMM_SUCCESS ||
        packetHandler->write1ByteTxRx(portHandler, CAMERA_ID, ADDR_OPERATING_MODE, 4, &dxl_error) != COMM_SUCCESS) {
      RCLCPP_ERROR(logger, "Camera: failed to set Extended Position Control Mode");
      return false;
    }
    uint32_t present = 0;
    if (packetHandler->read4ByteTxRx(portHandler, CAMERA_ID, ADDR_PRESENT_POSITION, &present, &dxl_error) != COMM_SUCCESS) {
      RCLCPP_ERROR(logger, "Camera: failed to read home position");
      return false;
    }
    camera_home_ticks = static_cast<int32_t>(present);
    if (packetHandler->write1ByteTxRx(portHandler, CAMERA_ID, ADDR_TORQUE_ENABLE, 1, &dxl_error) != COMM_SUCCESS) {
      RCLCPP_ERROR(logger, "Camera: failed to enable torque");
      return false;
    }
    camera_angle_rad = CAMERA_HOME_DEG * M_PI / 180.0;
    camera_ready = true;
    RCLCPP_INFO(logger, "Camera motor ready, home position: %d ticks", camera_home_ticks);
    return true;
  }

  void publish_joint_states(const rclcpp::Time & stamp, bool publish_camera)
  {
    auto make_arm_msg = [&](const std::array<int, 4> & idx) {
      sensor_msgs::msg::JointState js;
      js.header.stamp = stamp;
      js.name = {"inner_rotation", "outer_rotation", "inner_translation", "outer_translation"};
      for (int k : idx) {
        js.position.push_back(joint_phys[k]);
      }
      return js;
    };
    js_pub_right->publish(make_arm_msg(RIGHT_ARM_IDX));
    js_pub_left->publish(make_arm_msg(LEFT_ARM_IDX));

    if (publish_camera) {
      publish_camera_state(stamp);
    }
  }
}

//Main node that checks set-up of motors, initializes arrays, and establishes subscribers/publishers
MotorNode::MotorNode(const rclcpp::NodeOptions & options) : Node("motor_node",options) // Node = superclass, MotorNode = subclass
{
  RCLCPP_INFO(this->get_logger(), "Initializing motor node");

  // this->get_parameter("comments",comments);
  // if (comments) {
  //   RCLCPP_INFO(this->get_logger(),"Comments enabled.");
  // }

  //Dynamixel functions to write encoder positions to motors
  groupSyncWritePosition = new dynamixel::GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4);  
  ///CURRENT CONTROL////////////////////////////////////////////////
  groupSyncWriteCurrent = new dynamixel::GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_CURRENT, 2);
  groupSyncRead = new dynamixel::GroupSyncRead(portHandler,packetHandler, ADDR_PRESENT_POSITION, 4);

  /////Include if want to use discrete sections for camera control/////
  //current_state = MOVING;


  motor_exist = packetHandler->ping(portHandler,1,&dxl_error);
  if (motor_exist) {
  RCLCPP_INFO(this->get_logger(),"True");};

  //Check if 9th motor (gripper) is connected
  dxl_comm_result = packetHandler->ping(portHandler,9,&dxl_error);
  if (dxl_comm_result == COMM_SUCCESS) {
    has_gripper = true;
    RCLCPP_INFO(this->get_logger(),"Gripper motor detected");
  } else {
    has_gripper = false;
    RCLCPP_WARN(this->get_logger(),"Gripper motor not detected");
  }

  //Check if 10th motor (camera actuator) is connected
  // dxl_comm_result = packetHandler->ping(portHandler,10,&dxl_error);
  // if (dxl_comm_result == COMM_SUCCESS) {
  //   has_camera = true;
  //   RCLCPP_INFO(this->get_logger(),"Camera motor detected");
  // } else {
  //   has_camera = false;
  //   RCLCPP_WARN(this->get_logger(),"Camera motor not detected");
  // }
  has_camera = false;

  //Set motor array to appropriate size based on if gripper/camera motors are attached
  if (has_gripper && !has_camera) {
    motor_array.data = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    //float size = motor_array.data.size();
    //RCLCPP_INFO(this->get_logger(),"motor array size (g): %d",size);
  } else if (has_camera) {
    motor_array.data = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    //float size = motor_array.data.size();
    //RCLCPP_INFO(this->get_logger(),"motor array size (g&c): %d",size);
  };

  // Listening for directions for motors
  motor_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/robot/state/current_state", 10, std::bind(&MotorNode::position_callback, this, std::placeholders::_1));

  // Listening for keyboard input to manually control each motor with keyboard (adds to offset)
  keyboard_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/keyboard_control", 10, std::bind(&MotorNode::keyboard_callback, this, std::placeholders::_1));

  // Listening for "{data: 1}" for user to set new home position
  set_home_subscriber = this->create_subscription<std_msgs::msg::Int32>(
    "/set_home",10, std::bind(&MotorNode::set_home_callback,this,std::placeholders::_1));

  //Publishes camera position for tube tracking (data use only, not functional use)
  data_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/output_data_camera",10);
  //Sends motors home when home command called
  home_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/robot/state/current_state",10);
  //Calculates fwkin from joint values and publishes (x,y,z) coordinates
  fwkin_publisher_r = this->create_publisher<geometry_msgs::msg::Pose>("/fwkin_r",10);
  fwkin_publisher_l = this->create_publisher<geometry_msgs::msg::Pose>("/fwkin_l",10);

  /////Include if want to write camera data to csv/////
  // write_to_csv_subscription = this->create_subscription<std_msgs::msg::Float32MultiArray>("/output_data_camera",10,std::bind(&MotorNode::csv_callback,this,std::placeholders::_1));
  // csv_file.open("output_data_camera.csv",std::ios::out | std::ios::trunc);
  // if (csv_file.is_open()) {
  //   csv_file << "Timestamp,Data\n";
  // } else {
  //   RCLCPP_ERROR(this->get_logger(),"Failed to open CSV file");
  // }

  //Listening for directions to control 9th motor (gripper) only if gripper motor is connected (ID 9)
  if (has_gripper || has_camera) {
    gripper_sub = this->create_subscription<std_msgs::msg::Float32>(
      "/gripper_control",10,std::bind(&MotorNode::gripper_callback,this,std::placeholders::_1));
  };

  //Establish gear ratio
  if (this->has_parameter("gear_ratio")) {
    gear_ratio = this->get_parameter("gear_ratio").as_double();
  } else {
    this->declare_parameter("gear_ratio",1.0);  
    gear_ratio = this->get_parameter("gear_ratio").as_double();
  }

  if (comments) {
    RCLCPP_INFO(this->get_logger(), "Gear ratio: %.2f",gear_ratio);
  }

  /////Include if want to use discrete sections for camera control/////
  // timer_ptr = this->create_wall_timer(
  //   std::chrono::milliseconds(100),
  //   std::bind(&MotorNode::state_machine,this)
  // );

  //Initialize motor array for 8 motors as float 0s
  motor_array.data.assign(10, 0.0f);
  fwkin_xyz_l.position.x = 0.0;
  fwkin_xyz_l.position.y = 0.0;
  fwkin_xyz_l.position.z = 0.0;
  fwkin_xyz_l.orientation.x = 0.0;
  fwkin_xyz_l.orientation.y = 0.0;
  fwkin_xyz_l.orientation.z = 0.0;
  fwkin_xyz_l.orientation.w = 1.0;

  fwkin_xyz_r.position.x = 0.0;
  fwkin_xyz_r.position.y = 0.0;
  fwkin_xyz_r.position.z = 0.0;
  fwkin_xyz_r.orientation.x = 0.0;
  fwkin_xyz_r.orientation.y = 0.0;
  fwkin_xyz_r.orientation.z = 0.0;
  fwkin_xyz_r.orientation.w = 1.0;

  //Initialize camera position data (tube tracking)
  data_store.data = {0.0};
  //motor_array.assign(9,0.0);

//////////////////////////////////////////////////////////////////////////////////////
  // smoother_uterus integration: joint state publishers (see top of file)
  //////////////////////////////////////////////////////////////////////////////////////
  // Params may already exist via automatically_declare_parameters_from_overrides
  auto param_or = [this](const std::string & name, auto default_value) {
    if (!this->has_parameter(name)) {
      this->declare_parameter(name, default_value);
    }
    return this->get_parameter(name).get_value<decltype(default_value)>();
  };
  joint_readback     = param_or("joint_readback", false);
  double joint_state_rate_hz = param_or("joint_state_rate_hz", 30.0);

  js_pub_right  = this->create_publisher<sensor_msgs::msg::JointState>("/robot/right/joint/measured_jp", 10);
  js_pub_left   = this->create_publisher<sensor_msgs::msg::JointState>("/robot/left/joint/measured_jp", 10);
  js_pub_camera = this->create_publisher<sensor_msgs::msg::JointState>("/robot/camera/joint/measured_jp", 10);

  RCLCPP_INFO(this->get_logger(), "Joint states: %s at %.1f Hz",
    joint_readback ? "motor read-back" : "last command", joint_state_rate_hz);

   // Default callback group, same as position_callback and js_timer -> serialized bus access.
  camera_sub = this->create_subscription<std_msgs::msg::Float32>(
    "/motor_angle", 10,
    [this](const std_msgs::msg::Float32::SharedPtr msg) {
      if (!camera_ready) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
          "Camera motor not ready, ignoring /motor_angle");
        return;
      }

      // Safety limits
      float angle = std::clamp(msg->data, CAMERA_MIN_DEG, CAMERA_MAX_DEG);
      if (angle != msg->data) {
        RCLCPP_WARN(this->get_logger(),
          "Camera angle %.1f outside of range. Moving to %.1f degrees.", msg->data, angle);
      }

      // Degrees relative to home -> encoder ticks
      int32_t goal = camera_home_ticks +
        static_cast<int32_t>((angle - CAMERA_HOME_DEG) * CAMERA_TICKS_PER_DEG);

      if (packetHandler->write4ByteTxRx(portHandler, CAMERA_ID, ADDR_GOAL_POSITION,
            static_cast<uint32_t>(goal), &dxl_error) != COMM_SUCCESS) {
        RCLCPP_ERROR(this->get_logger(), "Camera: failed to send position command");
        return;
      }

      camera_angle_rad = angle * M_PI / 180.0;
      publish_camera_state(this->now());
    });


  // Default (mutually exclusive) callback group: never runs concurrently with position_callback
  // or set_home_callback, so the serial port is never accessed from two threads at once.
  js_timer = this->create_wall_timer(
    std::chrono::nanoseconds(static_cast<int64_t>(1e9 / joint_state_rate_hz)),
    [this]() {
      if (joint_readback) {
        // Read present positions of the 8 arm motors and invert the command mapping used in
        // position_callback:
        //   rot_ticks   = q_rot * TICKS_PER_RAD * gear + home_rot
        //   trans_ticks = q_trans * TICKS_PER_M - q_rot * TICKS_PER_RAD * gear + home_trans
        // The camera angle always comes from the last command (unloaded, so command ~= actual).
        // At 115200 baud this read costs ~15 ms of bus time; keep the rate modest or raise the baud.
        groupSyncRead->clearParam();
        for (uint8_t id = 1; id <= 8; ++id) {
          groupSyncRead->addParam(id);
        }
        if (groupSyncRead->txRxPacket() != COMM_SUCCESS) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Joint read-back: GroupSyncRead failed, not publishing joint states");
          return;
        }
        for (int i = 0; i < 8; i += 2) {  // (translation motor ID i+1, rotation motor ID i+2)
          if (!groupSyncRead->isAvailable(i + 1, ADDR_PRESENT_POSITION, 4) ||
              !groupSyncRead->isAvailable(i + 2, ADDR_PRESENT_POSITION, 4)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
              "Joint read-back: missing data for motors %d/%d", i + 1, i + 2);
            return;
          }
          int32_t trans_ticks = static_cast<int32_t>(groupSyncRead->getData(i + 1, ADDR_PRESENT_POSITION, 4));
          int32_t rot_ticks   = static_cast<int32_t>(groupSyncRead->getData(i + 2, ADDR_PRESENT_POSITION, 4));
          double rot_rel = static_cast<double>(rot_ticks - motor_home[i + 1]);
          joint_phys[i + 1] = rot_rel / (TICKS_PER_RAD * gear_ratio);
          joint_phys[i]     = (static_cast<double>(trans_ticks - motor_home[i]) + rot_rel) / TICKS_PER_M;
        }
        have_joint_phys = true;
      }

      // Command mode: republish the last command with a fresh stamp (joints are static while idle)
      if (have_joint_phys) {
        publish_joint_states(this->now(), camera_ready);
      }
    });
}

//Callback function to control gripper motor. Up arrow opens grasper, down arrow closes grasper. 8mm translation is full open/close
void MotorNode::gripper_callback(const std_msgs::msg::Float32::SharedPtr msg) {

  


  // if (std::abs(msg->data - 0.008) < tolerance) {
  //       if (has_gripper) {
  //         motor_array.data[8] = 0.008;
  //         RCLCPP_INFO(this->get_logger(),"Gripper opened");
  //       };
  //     } else if (msg->data < 0 && std::abs(msg->data + 0.008) < tolerance) {
  //       if (has_gripper) {
  //         motor_array.data[8] = 0.0;
  //         RCLCPP_INFO(this->get_logger(),"Gripper closed");
  //       };
  //     };
        //Include if using discrete sections for camera control. Keyboard input of 1, 2, or 3 controls camera section
      // } else if (msg->data == 1.0 || msg->data == 2.0 || msg->data == 3.0) {
      //   RCLCPP_INFO(this->get_logger(),"Received camera adjust signal: %1f. Transitioning to CAMERA_ADJUST state.",msg->data);
      //   gripper_value = msg->data;
      //   current_state = CAMERA_ADJUST;
      //   //state_machine();
      //   //RCLCPP_INFO(this->get_logger(),"Tried to switch states");
      // }
}
//Callback function to allow for simultaneous keyboard control & other control (this adds to offset)
void MotorNode::keyboard_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
  for (int i = 0; i < 8; i++) {
    offset[i] += msg->data[i];
    // RCLCPP_INFO(this->get_logger(),"Offset %d: %f",i+1,msg->data[i]);
    // RCLCPP_INFO(this->get_logger(),"Offset %d: %f",i+1,offset[i]);
  }
}

//Homing function. Is called if 1 is sent along set_home topic
void MotorNode::setHomePos() {

  //Clear previously stored home position
  groupSyncRead->clearParam();

  // //Set # of loops for homing based on how many motors are connected
  // if (has_gripper && has_camera) {
  //   loops = 10;
  //   //RCLCPP_INFO(this->get_logger(),"loops: %d",loops);
  // } else if (has_gripper && !has_camera) {
  //   loops = 9;
  // } else if (!has_gripper && !has_camera) {
  //   loops = 8;
  // }
  loops = 8;

  for (int i = 0; i < loops; i++) {
    //RCLCPP_INFO(this->get_logger(),"loop: %d",i);
    while (!groupSyncRead->addParam(i+1)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to add motor ID %d to GroupSyncRead",i+1);
      //return;
    }
    //RCLCPP_INFO(this->get_logger(),"Added motor to GroupSyncRead");
  }

  //If can't read all motor position (motors not on/connected). Note: while loop so function will wait until connection is re-established
  if (groupSyncRead->txRxPacket() != COMM_SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Failed to GroupSyncRead");
    //return;
  }
  RCLCPP_INFO(this->get_logger(),"Motors ready to run!");
  std_msgs::msg::Float32MultiArray home_msg;
  home_msg.data.resize(9);
  for (int i = 0; i < loops; i+=2) {
    if (groupSyncRead->isAvailable(i+1,ADDR_PRESENT_POSITION,4)) {
      //Retrieve current motor position for each motor
      int32_t trans_motor_pos = static_cast<int32_t>(groupSyncRead->getData(i+1,ADDR_PRESENT_POSITION,4));
      int32_t rot_motor_pos = static_cast<int32_t>(groupSyncRead->getData(i+2,ADDR_PRESENT_POSITION,4));
      //Set rotation motor home to be the closest 360degree rotation to current position (less than current position)
      int32_t remainder = rot_motor_pos % ENCODER_TICKS_PER_REV;
      if ((remainder != 0) && (remainder > (4096/2))) {
        // motor_home[i+1] = rot_motor_pos - (rot_motor_pos % ENCODER_TICKS_PER_REV) + 4096;
        motor_home[i+1] = rot_motor_pos - remainder + 4096;
        RCLCPP_INFO(this->get_logger(), "Rotation home: %d",motor_home[i+1]);
        //Set translation motor home to be opposite of rotation home
        motor_home[i] = remainder - rot_motor_pos - 4096;
        RCLCPP_INFO(this->get_logger(), "Translation home: %d",motor_home[i]);
      } else {
        motor_home[i+1] = rot_motor_pos - remainder;
        // RCLCPP_INFO(this->get_logger(), "Rotation home: %d",motor_home[i+1]);
        //Set translation motor home to be opposite of rotation home
        motor_home[i] = remainder - rot_motor_pos;
        // RCLCPP_INFO(this->get_logger(), "Translation home: %d",motor_home[i]);
      }
    }
    else {
      RCLCPP_ERROR(this->get_logger(), "Failed to get position for motor %d",i+1);
    }

  }
  // RCLCPP_INFO(this->get_logger(),"Inner trans: %d",motor_home[6]);
  // RCLCPP_INFO(this->get_logger(),"Outer trans: %d",motor_home[4]);
  //checks to make sure inner tube translation is always last tube moving to avoid retraction inside of outer tube
  // if change outer trans > change inner trans -> add extra 360deg rotation to inner tube
  //outer trans = 0, 4 inner trans = 2, 6 inner rot = 3, 7
  for (int i = 0; i < 5; i+=4) {
    int32_t outer_trans_motor_pos = static_cast<int32_t>(groupSyncRead->getData(i+1,ADDR_PRESENT_POSITION,4));
    int32_t inner_trans_motor_pos = static_cast<int32_t>(groupSyncRead->getData(i+3,ADDR_PRESENT_POSITION,4));
    while ((abs((motor_home[i] - outer_trans_motor_pos))) > (abs((motor_home[i+2] - inner_trans_motor_pos)))) {  //if outer tube is going to move more than inner tube (causing inner tube to over-retract)
      motor_home[i+2] -= 4096;  //sub a rotation to inner trans (moves opposite rotary motor)
      motor_home[i+3] += 4096;  //add a rotation to inner rot
    }
  }

  // for (int i = 0; i < 8; i++)  {
  //   RCLCPP_INFO(this->get_logger(),"Motor home %d: %d",i+1,motor_home[i]);
  // }
  // RCLCPP_INFO(this->get_logger(),"Motor home: %f",motor_home[2]);  //should be 3830
  // std_msgs::msg::Float32MultiArray home_msg;
  // home_msg.data.resize(9);
  for (int i = 0; i < 8; i++) {
    home_msg.data[i] = static_cast<float>(0);
}
  offset = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  //Send motors to new home
  home_publisher->publish(home_msg);

  //RCLCPP_INFO(this->get_logger(),"New home pos %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",motor_home[0],motor_home[1],motor_home[2],motor_home[3],motor_home[4],motor_home[5],motor_home[6],motor_home[7]);

  //If gripper & camera motor connected
  if (has_camera) {
    if (first_home_loop) {    //if camera home has not been set (only happens upon start-up, not each time robot is homed)
      for (int i = 0; i < 10; i++) {
        first_motor_home[i] = motor_home[i];
      }
      first_home_loop = false;
      RCLCPP_INFO(this->get_logger(),"Set start-up home position: %.2d",motor_home[9]);
    }

  }

  RCLCPP_INFO(this->get_logger(),"Set home position!");

}

//Send ros2 topic pub -1 /set_home "{data: 1}" to set home position
void MotorNode::set_home_callback(const std_msgs::msg::Int32::SharedPtr msg) {
  if (msg->data == 1) {
    RCLCPP_INFO(this->get_logger(), "Received set home command.");
    setHomePos();
  } 
  else if (msg->data == 2){
    RCLCPP_INFO(this->get_logger(), "Sending motors to absolute 0");
    motor_home = {0,0,0,0,0,0,0,0};
    std_msgs::msg::Float32MultiArray home_msg;
    home_msg.data.resize(9);
    for (int i = 0; i < 8; i++) {
      home_msg.data[i] = static_cast<float>(0);
    } 
  //Send motors to absolute home
  home_publisher->publish(home_msg);
  }
  else {
    RCLCPP_INFO(this->get_logger(), "Received set home command, but value is not 1. Ignoring.");
  }
}


//Callback function to control motors. Bulk of work happens here.
void MotorNode::position_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  // moveMotors(msg->data); 
  //Clear previous motor positions
  groupSyncWritePosition->clearParam();
  groupSyncWriteCurrent->clearParam();

  //If camera & gripper motors are connected, perform automatic tube tracking
  if (has_camera) {
    loops = 10;
    //Set camera motor to be at average of left and right tube translation
    motor_array.data[9] = ((msg->data[2]+msg->data[6])/2 * scalar * REV_PER_RADIAN * ENCODER_TICKS_PER_REV);
    //If command is to pull tubes inside endoscope tip (doesn't happen), set camera position to 0 to avoid breaking camera
    if (motor_array.data[9] < 0) {
      motor_array.data[9] = 0;
    }
    else if ((motor_array.data[9] / (REV_PER_RADIAN * ENCODER_TICKS_PER_REV)) > 1.291) {
      motor_array.data[9] = 1.291 * REV_PER_RADIAN * ENCODER_TICKS_PER_REV;
    }

  }

  //Map kinematics output to motor array (Inner Rot, Outer Rot, Inner Trans, Outer Trans -> Outer Trans, Outer Rot, Inner Trans, Inner Rot)
  //Right arm
  this->motor_array.data[0] = msg->data[3];
  this->motor_array.data[1] = msg->data[1];
  this->motor_array.data[2] = msg->data[2];
  this->motor_array.data[3] = msg->data[0];
  //Left arm
  this->motor_array.data[4] = msg->data[7];
  this->motor_array.data[5] = msg->data[5];
  this->motor_array.data[6] = msg->data[6];
  this->motor_array.data[7] = msg->data[4];
  double I_outer = M_PI/64.0 * (pow(OD_OUTER,4) - pow(ID_OUTER,4));
  double I_inner = M_PI/64.0 * (pow(OD_INNER,4) - pow(ID_INNER,4));
  double chi_right = (E*I_outer*CURVATURE_OUTER*cos(motor_array.data[1]) + E*I_inner*CURVATURE_INNER*cos(motor_array.data[3]))/(E*I_outer + E*I_inner);
  double chi_left = (E*I_outer*CURVATURE_OUTER*cos(motor_array.data[5]) + E*I_inner*CURVATURE_INNER*cos(motor_array.data[7]))/(E*I_outer + E*I_inner);
  double gamma_right = (E*I_outer*CURVATURE_OUTER*sin(motor_array.data[1]) + E*I_inner*CURVATURE_INNER*sin(motor_array.data[3]))/(E*I_outer + E*I_inner);
  double gamma_left = (E*I_outer*CURVATURE_OUTER*sin(motor_array.data[5]) + E*I_inner*CURVATURE_INNER*sin(motor_array.data[7]))/(E*I_outer + E*I_inner);
  double K_right = sqrt(pow(chi_right,2) + pow(gamma_right,2));
  double K_left = sqrt(pow(chi_left,2) + pow(gamma_left,2));
  double phi_right = atan2(gamma_right,chi_right);
  double phi_left = atan2(gamma_left,chi_left);
  //xyz coordinates of right arm
  fwkin_xyz_r.position.x = ((1-cos(K_right*motor_array.data[0]))/K_right)*cos(phi_right) + (motor_array.data[2]-motor_array.data[0])*sin(K_right*motor_array.data[0])*cos(phi_right);
  fwkin_xyz_r.position.y = ((1-cos(K_right*motor_array.data[0]))/K_right)*sin(phi_right) + (motor_array.data[2]-motor_array.data[0])*sin(K_right*motor_array.data[0])*sin(phi_right);
  fwkin_xyz_r.position.z = (sin(K_right*motor_array.data[0])/K_right) + (motor_array.data[2]-motor_array.data[0])*cos(K_right*motor_array.data[0]);
  fwkin_publisher_r->publish(fwkin_xyz_r);
  //xyz coordinates of left arm
  fwkin_xyz_l.position.x = ((1-cos(K_left*motor_array.data[4]))/K_left)*cos(phi_left) + (motor_array.data[6]-motor_array.data[4])*sin(K_left*motor_array.data[4])*cos(phi_left);
  fwkin_xyz_l.position.y = ((1-cos(K_left*motor_array.data[4]))/K_left)*sin(phi_left) + (motor_array.data[6]-motor_array.data[4])*sin(K_left*motor_array.data[4])*sin(phi_left);
  fwkin_xyz_l.position.z = (sin(K_left*motor_array.data[4])/K_left) + (motor_array.data[6]-motor_array.data[4])*cos(K_left*motor_array.data[4]);
  fwkin_publisher_l->publish(fwkin_xyz_l);

  //RCLCPP_INFO(this->get_logger(),"Received: %.2f",motor_array.data[1]);

  //Calculate motor encoder positions from meters or radians and re-format to how motors receive info
  for (int i = 0; i<loops; i+=2)
  {
    //RCLCPP_INFO(this->get_logger(),"i: %d",i);
    //Motor position calculation
    // if (i !=8) {
    //   pos = (int32_t)(motor_array.data[i]*(double)ENCODER_TICKS_PER_REV*(double)LEAD_SCREW_REV_PER_M) + motor_home[i];
    //   //if (comments) {
    //     RCLCPP_INFO(this->get_logger(),"Motor ID: %d; Position (mm): %.3f",i+1,motor_array.data[i]*1000);
    //     RCLCPP_INFO(this->get_logger(),"Motor Home: %.2d",motor_home[i]);
    //   //Don't let outer tube retract past motor home position
    //   // if (i==0 | i==4) {
    //   //   if (pos < motor_home[i]) {
    //   //     pos = motor_home[i];
    //   //     RCLCPP_INFO(this->get_logger(),"Don't retract outer tube too far!");
    //   //   }
    //   // }
    //   //}

    //   //Separate by bytes and words for motors
    //   param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(pos));
    //   param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(pos));
    //   param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(pos));
    //   param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(pos));

    //   size_t param_size_pos = sizeof(param_goal_position)/ sizeof(param_goal_position[0]);

    // };

    // RCLCPP_INFO(this->get_logger(),"Offset %d: %f",i+1,offset[i]);
    // RCLCPP_INFO(this->get_logger(),"Offset %d: %f",i+1,offset[i+1]);

    //Set hard limits
    //Outer tube translation
    if ((i == 0) | (i == 4)) {
      if ((motor_array.data[i] + offset[i]) > 0.04) {
      // if (motor_array.data[i] > 0.0495) {
        motor_array.data[i] = 0.04 - offset[i];
        // motor_array.data[i] = 0.0495;
        RCLCPP_INFO(this->get_logger(),"Reached outer tube limit!");
      }
    }
    //Inner tube translation
    if ((i == 2) | (i == 6)) {
      if ((motor_array.data[i] + offset[i]) > 0.09) { 
        RCLCPP_INFO(this->get_logger(),"Offset: %f",motor_array.data[i]);
      // if ((motor_array.data[i]) > 0.105) {
        motor_array.data[i] = 0.09 - offset[i];
        // motor_array.data[i] = 0.105;
        RCLCPP_INFO(this->get_logger(),"Reached inner tube limit!");
      }
      // if ((motor_array.data[i] + offset[i]) < (motor_array.data[i-2] + offset[i-2])) {
      // if ((motor_array.data[i]) < (motor_array.data[i-2])) {
      if ((motor_array.data[i] + offset[i]) < 0.0) {
        // RCLCPP_INFO(this->get_logger(),"input: %f",(motor_array.data[i] + offset[i]));
        // RCLCPP_INFO(this->get_logger(),"compare: %f",(motor_array.data[i-2] + offset[i-2]));
        // RCLCPP_INFO(this->get_logger(),"set: %f",m);
        // override = motor_array.data[i-2] + offset[i-2];
        RCLCPP_INFO(this->get_logger(),"Don't retract inner tube too far!");
        // override = motor_array.data[i-2];
        override = motor_array.data[i-2] - offset[i];
        // motor_array.data[i] = override - offset[i];
        motor_array.data[i] = override;
        // RCLCPP_INFO(this->get_logger(),"set: %f",motor_array.data[i]);
        // RCLCPP_INFO(this->get_logger(),"set: %f",override);
        // RCLCPP_INFO(this->get_logger(),"Don't retract inner tube inside outer tube!");
      }
    }

    //Motor rotation calculation
    if (i != 8) {
      //rot(encoder ticks) = desired rot (encoder ticks) + motor home (encoder ticks) + rot offset from keyboard (encoder ticks)
      rot = (int32_t)(motor_array.data[i+1]*(double)ENCODER_TICKS_PER_REV*(double)REV_PER_RADIAN*gear_ratio) + motor_home[i+1] + (int32_t)(offset[i+1]*(double)ENCODER_TICKS_PER_REV*(double)REV_PER_RADIAN*gear_ratio);
    } else {
      //rot = (int32_t)(motor_array.data[i+1]*(double)ENCODER_TICKS_PER_REV_CAMERA*(double)REV_PER_RADIAN) + motor_home[i+1];
      rot = motor_array.data[i+1];
      //RCLCPP_INFO(this->get_logger(),"camera: %.2f",rot);
    }
    if (comments) {
        RCLCPP_INFO(this->get_logger(),"Motor ID: %d; Rotation (deg): %f",i+2,rot);
        RCLCPP_INFO(this->get_logger(),"Motor Home: %d",motor_home[i+1]);
      }
    
    //Separate by bytes and words for motors
    param_goal_rotation[0] = DXL_LOBYTE(DXL_LOWORD(rot));
    param_goal_rotation[1] = DXL_HIBYTE(DXL_LOWORD(rot));
    param_goal_rotation[2] = DXL_LOBYTE(DXL_HIWORD(rot));
    param_goal_rotation[3] = DXL_HIBYTE(DXL_HIWORD(rot));

    //current control
    // int16_t goal_current = 1;  // Example: 300 * 2.69 mA = ~807 mA
    ///CURRENT CONTROL////////////////////////////////////////////////
    uint8_t param_goal_current[2];
    // RCLCPP_INFO(this->get_logger(), "Goal current: %d (approx %.2f mA)", goal_current, goal_current * 2.69);
    ///CURRENT CONTROL////////////////////////////////////////////////
    param_goal_current[0] = DXL_LOBYTE(goal_current);
    param_goal_current[1] = DXL_HIBYTE(goal_current);

    //size_t param_size_rot = sizeof(param_goal_rotation)/sizeof(param_goal_rotation[0]);

    if (i !=8) {
      //pos(encoder ticks) = desired pos (encoder ticks) - desired rotation (encoder ticks) + motor home (encoder ticks) + trans offset from keyboard (encoder ticks) - rot offset from keyboard (encoder ticks)
      pos = (int32_t)(motor_array.data[i]*(double)ENCODER_TICKS_PER_REV*(double)LEAD_SCREW_REV_PER_M) - (int32_t)(motor_array.data[i+1]*(double)ENCODER_TICKS_PER_REV*(double)REV_PER_RADIAN*gear_ratio) + motor_home[i] + 
        (int32_t)(offset[i]*(double)ENCODER_TICKS_PER_REV*(double)LEAD_SCREW_REV_PER_M) - (int32_t)(offset[i+1]*(double)ENCODER_TICKS_PER_REV*(double)REV_PER_RADIAN*gear_ratio);
      if (comments) {
        RCLCPP_INFO(this->get_logger(),"Motor ID: %d; Position (mm): %f",i+1,pos);
        RCLCPP_INFO(this->get_logger(),"Motor Home: %d",motor_home[i]);
      //Don't let outer tube retract past motor home position
      // if (i==0 | i==4) {
      //   if (pos < motor_home[i]) {
      //     pos = motor_home[i];
      //     RCLCPP_INFO(this->get_logger(),"Don't retract outer tube too far!");
      //   }
      // }
      }

      //Separate by bytes and words for motors
      param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(pos));
      param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(pos));
      param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(pos));
      param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(pos));

      ///CURRENT CONTROL////////////////////////////////////////////////
      param_goal_current[0] = DXL_LOBYTE(goal_current);
      param_goal_current[1] = DXL_HIBYTE(goal_current);

      //size_t param_size_pos = sizeof(param_goal_position)/ sizeof(param_goal_position[0]);

    };

    //Make sure motors were added to groupSyncWrite (able to write encoder positions to motors)
    if (i != 8) {
      ///CURRENT CONTROL////////////////////////////////////////////////
      bool add_success_current = groupSyncWriteCurrent->addParam(i+1, param_goal_current);
      if (!add_success_current) {
        RCLCPP_ERROR(this->get_logger(), "Failed to limit current");
      }
      bool add_success1 = groupSyncWritePosition->addParam(i+1, param_goal_position);
      if (!add_success1) {
        RCLCPP_ERROR(this->get_logger(), "Failed to add position pos");
      }
      // bool add_success_current = groupSyncWriteCurrent->addParam(i+1, param_goal_current);
      // if (!add_success_current) {
      //   RCLCPP_ERROR(this->get_logger(), "Failed to limit current");
      // }
    }

    ///CURRENT CONTROL////////////////////////////////////////////////
    bool add_success_current = groupSyncWriteCurrent->addParam(i+2, param_goal_current);
    if (!add_success_current) {
      RCLCPP_ERROR(this->get_logger(), "Failed to limit current");
    }
    bool add_success2 = groupSyncWritePosition->addParam(i+2, param_goal_rotation);
    if (!add_success2) {
      RCLCPP_ERROR(this->get_logger(), "Failed to add rotation pos");
    } 
    // if (groupSyncWriteCurrent->txPacket() != COMM_SUCCESS) {
    //   RCLCPP_ERROR(this->get_logger(), "Failed to send current values: %s", packetHandler->getTxRxResult(dxl_comm_result));
    // }
    // groupSyncWriteCurrent->clearParam();
    // bool add_success_current = groupSyncWriteCurrent->addParam(i+2, param_goal_current);
    // if (!add_success_current) {
    //   RCLCPP_ERROR(this->get_logger(), "Failed to limit current");
    // }
  }

  if (groupSyncWriteCurrent->txPacket() != COMM_SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Failed to send current values: %s", packetHandler->getTxRxResult(dxl_comm_result));
  }
  groupSyncWriteCurrent->clearParam();

  // //Move gripper motor to appropriate location (open or closed) - handled separately from 8 motor control 
  // double gripper = (int32_t)(motor_array.data[8] * (double)ENCODER_TICKS_PER_REV * (double)LEAD_SCREW_REV_PER_M) + motor_home[8];
  // param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(gripper));
  // param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(gripper));
  // param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(gripper));
  // param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(gripper));
  
  // bool add_success9 = groupSyncWritePosition->addParam(9,param_goal_position);
  // if (!add_success9) {
  //   RCLCPP_ERROR(this->get_logger(),"Failed to add motor 9");}

  if (!at_rest) {
  dxl_comm_result = groupSyncWritePosition->txPacket();
  if (groupSyncWritePosition->txPacket() != COMM_SUCCESS)
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to get position: %s",packetHandler->getTxRxResult(dxl_comm_result));
  }
  }

  //Clear motors
  groupSyncRead->clearParam();
  //If camera motor is connected
  if (has_camera) {
    while (!groupSyncRead->addParam(10)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to add motor ID %d to GroupSyncRead",10);
      }
    if (groupSyncRead->txRxPacket() != COMM_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to GroupSyncRead");
    }
    if (groupSyncRead->isAvailable(10,ADDR_PRESENT_POSITION,4)) {
        //Retrieve current motor position for each motor
        camera_pos = groupSyncRead->getData(10,ADDR_PRESENT_POSITION,4);
      }
    else {
      RCLCPP_ERROR(this->get_logger(), "Failed to get position for motor %d",10);
    }

    // RCLCPP_INFO(this->get_logger(),"camera pos: %.2f",camera_pos);

    data_store.data[0] = camera_pos;
    
    data_publisher->publish(data_store);
  }

  // smoother_uterus integration: in command mode, the physical joints are what was just sent
  // (post-limit command + keyboard offset). Publish now; the timer republishes while idle.
  //camera_angle_rad = has_camera ? motor_array.data[9] / TICKS_PER_RAD : 0.0;
  if (!joint_readback) {
    for (int i = 0; i < 8; i++) {
      joint_phys[i] = motor_array.data[i] + offset[i];
    }
    have_joint_phys = true;
    publish_joint_states(this->now(), camera_ready);
  }
  
};

//Motor setup function
void setupDynamixel(uint8_t dxl_id, bool comments)
{
  // Use Extended Position Control Mode (4) or Extended Position Current Control Mode (5)
  dxl_comm_result = packetHandler->write1ByteTxRx(
      portHandler,
      dxl_id,
      ADDR_OPERATING_MODE,
      5,
      &dxl_error);
    
  // int16_t goal_current = 1;  // Example: 300 * 2.69 mA = ~807 mA
  //Set control mode
  if (dxl_comm_result != COMM_SUCCESS)
  {
    RCLCPP_ERROR(rclcpp::get_logger("motor_node"), "Failed to set Position Control Mode.");
  }
  else
  {
    if (comments) {
      RCLCPP_INFO(rclcpp::get_logger("motor_node"), "Succeeded to set Position Control Mode.");
    }
  }

  // Enable Torque of DYNAMIXEL
  dxl_comm_result = packetHandler->write1ByteTxRx(
      portHandler,
      dxl_id,
      ADDR_TORQUE_ENABLE,
      1,
      &dxl_error);

  // dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, dxl_id, 112, 200, &dxl_error); // Profile Velocity
  // dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, dxl_id, 108, 70, &dxl_error);  // Profile Acceleration

  //Enable torque
  if (dxl_comm_result != COMM_SUCCESS)
  {
    RCLCPP_ERROR(rclcpp::get_logger("motor_node"), "Failed to enable torque.");
  }
  else
  {
    if (comments) {
      RCLCPP_INFO(rclcpp::get_logger("motor_node"), "Succeeded to enable torque.");
    }
  }
}

//Main running function to constantly spin MotorNode
int main(int argc, char *argv[])
{

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true);

  portHandler = dynamixel::PortHandler::getPortHandler(DEVICE_NAME);
  packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

  auto motornode = std::make_shared<MotorNode>(options);
  
  bool comments = true;
  motornode->get_parameter("comments",comments);
  if (comments) {
    RCLCPP_INFO(motornode->get_logger(),"Comments enabled.");
  }


  // Open Serial Port
  dxl_comm_result = portHandler->openPort();
  if (dxl_comm_result == false)
  {
    RCLCPP_ERROR(rclcpp::get_logger("motor_node"), "Failed to open the port! :(");
    return -1;
  }
  else
  {
    if (comments) {
      RCLCPP_INFO(rclcpp::get_logger("motor_write_node"), "Succeeded to open the port.");
    }
  }


  // Set the baudrate of the serial port (use DYNAMIXEL Baudrate)
  dxl_comm_result = portHandler->setBaudRate(BAUDRATE);
  if (dxl_comm_result == false)
  {
    RCLCPP_ERROR(rclcpp::get_logger("motor_node"), "Failed to set the baudrate!");
    return -1;
  }
  else
  {
    if (comments) {
      RCLCPP_INFO(rclcpp::get_logger("motor_node"), "Succeeded to set the baudrate.");
    }
  }

  setupDynamixel(BROADCAST_ID, comments);

  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 1, 112, 176, &dxl_error); // Profile Velocity
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 1, 108, 70, &dxl_error);  // Profile Acceleration
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 2, 112, 200, &dxl_error); // Profile Velocity
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 2, 108, 70, &dxl_error);  // Profile Acceleration
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 3, 112, 176, &dxl_error); // Profile Velocity
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 3, 108, 70, &dxl_error);  // Profile Acceleration
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 4, 112, 200, &dxl_error); // Profile Velocity
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 4, 108, 70, &dxl_error);  // Profile Acceleration
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 5, 112, 167, &dxl_error); // Profile Velocity
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 5, 108, 70, &dxl_error);  // Profile Acceleration
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 6, 112, 200, &dxl_error); // Profile Velocity
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 6, 108, 70, &dxl_error);  // Profile Acceleration
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 7, 112, 167, &dxl_error); // Profile Velocity
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 7, 108, 70, &dxl_error);  // Profile Acceleration
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 8, 112, 200, &dxl_error); // Profile Velocity
  dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, 8, 108, 70, &dxl_error);  // Profile Acceleration

  //Override broadcast setup for motor 10 and record its home
  setupCamera();

  //Set home position as current position when motor node is first initialized
  motornode->setHomePos();

  //MultiThreadedExecutor allows for processing multiple messages in parallel (necessary for camera control & data publishing with standard motor control)
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(motornode);
  executor.spin();

  // Release file-scope ROS handles before shutdown (avoid static destruction after rclcpp teardown)
  js_timer.reset();
  js_pub_right.reset();
  js_pub_left.reset();
  js_pub_camera.reset();
  camera_sub.reset();

  rclcpp::shutdown();
  if (motornode->csv_file.is_open()) {
    motornode->csv_file.close();
  }

  // Disable Torque of DYNAMIXEL
  packetHandler->write1ByteTxRx(
      portHandler,
      BROADCAST_ID,
      ADDR_TORQUE_ENABLE,
      1,
      &dxl_error);

  packetHandler->write1ByteTxRx(portHandler, CAMERA_ID, ADDR_TORQUE_ENABLE, 0, &dxl_error);
      

  return 0;
}