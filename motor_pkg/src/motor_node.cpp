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

#include <dynamixel_sdk/dynamixel_sdk.h>  //Dynamixel library

#include <rclcpp/rclcpp.hpp>
#include "rcutils/cmdline_parser.h"

#include "motor_node.hpp"
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
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
#define CURVATURE_OUTER 100
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
  dxl_comm_result = packetHandler->ping(portHandler,10,&dxl_error);
  if (dxl_comm_result == COMM_SUCCESS) {
    has_gripper = true;
    RCLCPP_INFO(this->get_logger(),"Gripper motor detected");
  } else {
    has_gripper = false;
    RCLCPP_WARN(this->get_logger(),"Gripper motor not detected");
  }

  //Check if 10th motor (camera actuator) is connected
  dxl_comm_result = packetHandler->ping(portHandler,9,&dxl_error);
  if (dxl_comm_result == COMM_SUCCESS) {
    has_camera = true;
    RCLCPP_INFO(this->get_logger(),"Camera motor detected");
  } else {
    has_camera = false;
    RCLCPP_WARN(this->get_logger(),"Camera motor not detected");
  }

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
  motor_array.data = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
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

}

/////Include if want to write camera position to csv/////
// void MotorNode::csv_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
//   auto now = std::chrono::system_clock::now();
//   auto duration = now.time_since_epoch();
//   double time_in_sec = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
//   std::string time_str = std::to_string(time_in_sec);
//   std::string data_str = "[";
//   for (size_t i = 0; i < msg->data.size(); ++i) {
//     data_str += std::to_string(msg->data[i]);
//     if (i < msg->data.size()-1) {
//       data_str += ", ";
//     }
//   }
//   data_str += "]";

//   if (csv_file.is_open()) {
//     csv_file << time_str << "," << data_str << "\n";
//   }
// }

/////Include if want to use discrete sections for camera control/////
// void MotorNode::state_machine() {
//   //RCLCPP_INFO(this->get_logger(),"Current state: %d",current_state);
//   switch(current_state) {
//     case MOVING:
//       //RCLCPP_INFO(this->get_logger(),"Motors in MOVING state.");
//       //rclcpp::spin_some(this->get_node_base_interface());
//       break;
//     case CAMERA_ADJUST:
//       //RCLCPP_INFO(this->get_logger(),"Motors in CAMERA_ADJUST state.");
//       camera_adjust();
//       break;
//     case ERROR:
//       RCLCPP_ERROR(this->get_logger(),"Motors in ERROR state.");
//       break;
//     default:
//       RCLCPP_ERROR(this->get_logger(),"Unknown state.");
//       break;
//   }
// }

//Include if want to use discrete sections for camera control/////
//Node to motorize camera to view discrete sections of camera rotating FOV
//Pressing 1, 2, or 3 will command camera to specified section of rotating FOV. Controllers will then not control the robot until homed again.
// void MotorNode::camera_adjust() {
//   //Clear motors
//   groupSyncWrite->clearParam();

//   //If in moving state instead of tolerance checking state
//   if (first_camera_loop) {
//     first_camera_loop = false;    //boolean to determine if camera should be moved or if controllers need to be homed

//     //Code copied from standard motor function
//     for (int i = 0; i < 10; i++) {
//         motor_home[i] = first_motor_home[i];
//         //RCLCPP_INFO(this->get_logger(),"Motor home (initial): %.2f",motor_home[i]);
//       }

//     //RCLCPP_INFO(this->get_logger(),"Home pos for camera adjust %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",motor_home[0],motor_home[1],motor_home[2],motor_home[3],motor_home[4],motor_home[5],motor_home[6],motor_home[7]);  
//     //RCLCPP_INFO(this->get_logger(),"First home pos %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",first_motor_home[0],first_motor_home[1],first_motor_home[2],first_motor_home[3],first_motor_home[4],first_motor_home[5],first_motor_home[6],first_motor_home[7]);

//     //Sets camera to specified section and tubes to center of new workspace
//     if (gripper_value == 1.0) {
//       motor_array.data = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
//       RCLCPP_INFO(this->get_logger(),"Set section 1");
//     } else if (gripper_value == 2.0) {
//       motor_array.data = {0.0,0.0,-0.006283,-0.006283,0.0,0.0,-0.006283,-0.006283,0.0,camera_s2};
//       RCLCPP_INFO(this->get_logger(),"Set section 2");
//     } else if (gripper_value == 3.0) {
//       motor_array.data = {0.0,0.0,-0.01117,-0.01117,0.0,0.0,-0.01117,-.01117,0.0,camera_s3};
//       RCLCPP_INFO(this->get_logger(),"Set section 3");
//     }

//     //Right arm
//     //add temporary placeholder
//     std_msgs::msg::Float32MultiArray placeholder_motor_array;
//     placeholder_motor_array.data = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
//     placeholder_motor_array.data[0] = motor_array.data[3];
//     placeholder_motor_array.data[1] = motor_array.data[1];
//     placeholder_motor_array.data[2] = motor_array.data[2];
//     placeholder_motor_array.data[3] = motor_array.data[0];
//     //Left arm
//     placeholder_motor_array.data[4] = motor_array.data[7];
//     placeholder_motor_array.data[5] = motor_array.data[5];
//     placeholder_motor_array.data[6] = motor_array.data[6];
//     placeholder_motor_array.data[7] = motor_array.data[4];
//     //Gripper & camera
//     placeholder_motor_array.data[8] = motor_array.data[8];
//     placeholder_motor_array.data[9] = motor_array.data[9];
//     motor_array.data = placeholder_motor_array.data;


//     for (int i = 0; i < 10; i+=2) {
//       //Motor position calculation
//       pos = (int32_t)(motor_array.data[i]*(double)ENCODER_TICKS_PER_REV*(double)LEAD_SCREW_REV_PER_M) + motor_home[i];

//       //RCLCPP_INFO(this->get_logger(),"Motor ID: %d; Position (mm): %.3f",i+1,motor_array.data[i]*1000);

//       //Separate by bytes and words for motors
//       param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(pos));
//       param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(pos));
//       param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(pos));
//       param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(pos));

//       size_t param_size_pos = sizeof(param_goal_position)/ sizeof(param_goal_position[0]);

//       //Motor rotation calculation
//       if (i != 8) {
//         rot = (int32_t)(motor_array.data[i+1]*(double)ENCODER_TICKS_PER_REV*(double)REV_PER_RADIAN) + motor_home[i+1];
//        //RCLCPP_INFO(this->get_logger(),"used regular rev/radian");
//       } else {
//         rot = (int32_t)(motor_array.data[i+1]*(double)ENCODER_TICKS_PER_REV_CAMERA*(double)REV_PER_RADIAN) + motor_home[i+1];
//         //RCLCPP_INFO(this->get_logger(),"used camera rev/radian");
//       };
      
//       //Separate by bytes and words for motors
//       param_goal_rotation[0] = DXL_LOBYTE(DXL_LOWORD(rot));
//       param_goal_rotation[1] = DXL_HIBYTE(DXL_LOWORD(rot));
//       param_goal_rotation[2] = DXL_LOBYTE(DXL_HIWORD(rot));
//       param_goal_rotation[3] = DXL_HIBYTE(DXL_HIWORD(rot));

//       size_t param_size_rot = sizeof(param_goal_rotation)/sizeof(param_goal_rotation[0]);

//       //RCLCPP_INFO(this->get_logger(),"Motor ID: %d; Rotation (deg): %.3f",i+2,motor_array.data[i+1]*(180/M_PI));

//       //Make sure motors were added to groupSyncWrite (able to write encoder positions to motors)
//       bool add_success1 = groupSyncWrite->addParam(i+1, param_goal_position);
//       bool add_success2 = groupSyncWrite->addParam(i+2, param_goal_rotation);

//       if (!add_success1) {
//         RCLCPP_ERROR(this->get_logger(), "Failed to add position cam");
//       }

//       if (!add_success2) {
//         RCLCPP_ERROR(this->get_logger(), "Failed to add rotation cam");};
//     };

//     //Send motors to new position
//     dxl_comm_result = groupSyncWrite->txPacket();
//     if (dxl_comm_result == COMM_SUCCESS) {
//       RCLCPP_INFO(this->get_logger(),"Sent new position to motors!");
//     } else {
//       RCLCPP_INFO(this->get_logger(),"Couldn't groupSyncWrite: %s",dxl_comm_result);
//     }

//   };

//   //After sending motors to new location, controllers must be homed before control is resumed. Check if controller input is homed (.2mm)
//   check = is_within_tolerance();
//   if (!check){
//     //RCLCPP_INFO(this->get_logger(),"Entered while loop");
//     at_rest = true;
//     check = is_within_tolerance();
//   } else if (check) {
//     //RCLCPP_INFO(this->get_logger(),"Exited while loop");
//     at_rest = false;
//     RCLCPP_WARN(this->get_logger(),"UI homed. Returning to MOVING state.");
//     setHomePos();
//     //RCLCPP_INFO(this->get_logger(),"Home pos after camera adjust: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",motor_home[0],motor_home[1],motor_home[2],motor_home[3],motor_home[4],motor_home[5],motor_home[6],motor_home[7]);
//     //Update state machine back to moving state
//     current_state = MOVING;
//     first_camera_loop = true;
//   };
//   //state_machine();

// }

//Include if using camera control with discrete sections
//Checks if controllers are within tolerance of home position
// bool MotorNode::is_within_tolerance() {
//   std::vector<float> target1 = {0.0,0.0,M_PI,0.0,0.0,M_PI,0.0002,0.0};
//   std::vector<float> target2 = {0.0,0.0,-M_PI,0.0,0.0,-M_PI,0.0002,0.0};
//   std::vector<float> target3 = {0.0,0.0,0.0,0.0,0.0,0,0.0002,0.0};
  
//   std::vector<float> current = last_received_position;
//   float tol = 0.003;
//   for (size_t i = 0; i < 8; ++i) {
//     if (std::abs((current[i] - target1[i]) > tol) && (std::abs(current[i] - target2[i]) > tol) && (std::abs(current[i] - target3[i]) > tol)) {
//       return false;
//     };
//   }
//   return true;
// };

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

  //Set # of loops for homing based on how many motors are connected
  if (has_gripper && has_camera) {
    loops = 10;
    //RCLCPP_INFO(this->get_logger(),"loops: %d",loops);
  } else if (has_gripper && !has_camera) {
    loops = 9;
  } else if (!has_gripper && !has_camera) {
    loops = 8;
  }

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
  RCLCPP_INFO(this->get_logger(),"Motors ready to run!");\
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

// void moveMotors(const std::vector<float>& positions)
// {
//   //Clear previous motor positions
//   groupSyncWrite->clearParam();

//   //If camera & gripper motors are connected, perform automatic tube tracking
//   if (has_camera) {
//     loops = 10;
//     //Set camera motor to be at average of left and right tube translation
//     motor_array.data[9] = ((positions[2]+positions[6])/2 * scalar * REV_PER_RADIAN * ENCODER_TICKS_PER_REV);
//     //If command is to pull tubes inside endoscope tip (doesn't happen), set camera position to 0 to avoid breaking camera
//     if (motor_array.data[9] < 0) {
//       motor_array.data[9] = 0;
//     }
//     else if ((motor_array.data[9] / (REV_PER_RADIAN * ENCODER_TICKS_PER_REV)) > 1.291) {
//       motor_array.data[9] = 1.291 * REV_PER_RADIAN * ENCODER_TICKS_PER_REV;
//     }

//   }

//   //Map kinematics output to motor array (Inner Rot, Outer Rot, Inner Trans, Outer Trans -> Outer Trans, Outer Rot, Inner Trans, Inner Rot)
//   //Right arm
//   this->motor_array.data[0] = positions[3];
//   this->motor_array.data[1] = positions[1];
//   this->motor_array.data[2] = positions[2];
//   this->motor_array.data[3] = positions[0];
//   //Left arm
//   this->motor_array.data[4] = positions[7];
//   this->motor_array.data[5] = positions[5];
//   this->motor_array.data[6] = positions[6];
//   this->motor_array.data[7] = positions[4];

//   //RCLCPP_INFO(this->get_logger(),"Received: %.2f",motor_array.data[1]);

//   //Calculate motor encoder positions from meters or radians and re-format to how motors receive info
//   for (int i = 0; i<loops; i+=2)
//   {
//     //RCLCPP_INFO(this->get_logger(),"i: %d",i);
//     //Motor position calculation
//     if (i !=8) {
//       pos = (int32_t)(motor_array.data[i]*(double)ENCODER_TICKS_PER_REV*(double)LEAD_SCREW_REV_PER_M) + motor_home[i];
//       if (comments) {
//         RCLCPP_INFO(this->get_logger(),"Motor ID: %d; Position (mm): %.3f",i+1,motor_array.data[i]*1000);
//         RCLCPP_INFO(this->get_logger(),"Motor Home: %.2f",motor_home[i]);
//       //Don't let outer tube retract past motor home position
//       // if (i==0 | i==4) {
//       //   if (pos < motor_home[i]) {
//       //     pos = motor_home[i];
//       //     RCLCPP_INFO(this->get_logger(),"Don't retract outer tube too far!");
//       //   }
//       // }
//       }

//       //Separate by bytes and words for motors
//       param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(pos));
//       param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(pos));
//       param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(pos));
//       param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(pos));

//       size_t param_size_pos = sizeof(param_goal_position)/ sizeof(param_goal_position[0]);

//     };

//     //Motor rotation calculation
//     if (i != 8) {
//       rot = (int32_t)(motor_array.data[i+1]*(double)ENCODER_TICKS_PER_REV*(double)REV_PER_RADIAN*gear_ratio) + motor_home[i+1];
//     } else {
//       //rot = (int32_t)(motor_array.data[i+1]*(double)ENCODER_TICKS_PER_REV_CAMERA*(double)REV_PER_RADIAN) + motor_home[i+1];
//       rot = motor_array.data[i+1];
//       //RCLCPP_INFO(this->get_logger(),"camera: %.2f",rot);
//     }
//     if (comments) {
//         RCLCPP_INFO(this->get_logger(),"Motor ID: %d; Rotation (deg): %.1f",i+2,motor_array.data[i+1]*180/M_PI);
//         RCLCPP_INFO(this->get_logger(),"Motor Home: %.2f",motor_home[i+1]);
//       }
    
//     //Separate by bytes and words for motors
//     param_goal_rotation[0] = DXL_LOBYTE(DXL_LOWORD(rot));
//     param_goal_rotation[1] = DXL_HIBYTE(DXL_LOWORD(rot));
//     param_goal_rotation[2] = DXL_LOBYTE(DXL_HIWORD(rot));
//     param_goal_rotation[3] = DXL_HIBYTE(DXL_HIWORD(rot));

//     size_t param_size_rot = sizeof(param_goal_rotation)/sizeof(param_goal_rotation[0]);

//     //Make sure motors were added to groupSyncWrite (able to write encoder positions to motors)
//     if (i != 8) {
//       bool add_success1 = groupSyncWrite->addParam(i+1, param_goal_position);
//       if (!add_success1) {
//         RCLCPP_ERROR(this->get_logger(), "Failed to add position pos");
//       }
//     }

//     bool add_success2 = groupSyncWrite->addParam(i+2, param_goal_rotation);

//     if (!add_success2) {
//       RCLCPP_ERROR(this->get_logger(), "Failed to add rotation pos");
//     } else if (add_success2 && i == 8) {
//     }
//   }

//   //Move gripper motor to appropriate location (open or closed) - handled separately from 8 motor control 
//   double gripper = (int32_t)(motor_array.data[8] * (double)ENCODER_TICKS_PER_REV * (double)LEAD_SCREW_REV_PER_M) + motor_home[8];
//   param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(gripper));
//   param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(gripper));
//   param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(gripper));
//   param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(gripper));
  
//   bool add_success9 = groupSyncWrite->addParam(9,param_goal_position);
//   if (!add_success9) {
//     RCLCPP_ERROR(this->get_logger(),"Failed to add motor 9");
//   }
//   if (!at_rest) {
//   dxl_comm_result = groupSyncWrite->txPacket();
//   if (groupSyncWrite->txPacket() != COMM_SUCCESS)
//   {
//     RCLCPP_ERROR(this->get_logger(), "Failed to get position: %s",packetHandler->getTxRxResult(dxl_comm_result));
//   }
//   }

//   //Clear motors
//   groupSyncRead->clearParam();
//   //If camera motor is connected
//   if (has_camera) {
//     while (!groupSyncRead->addParam(10)) {
//         RCLCPP_ERROR(this->get_logger(), "Failed to add motor ID %d to GroupSyncRead",10);
//       }
//     if (groupSyncRead->txRxPacket() != COMM_SUCCESS) {
//       RCLCPP_ERROR(this->get_logger(), "Failed to GroupSyncRead");
//     }
//     if (groupSyncRead->isAvailable(10,ADDR_PRESENT_POSITION,4)) {
//         //Retrieve current motor position for each motor
//         camera_pos = groupSyncRead->getData(10,ADDR_PRESENT_POSITION,4);
//       }
//     else {
//       RCLCPP_ERROR(this->get_logger(), "Failed to get position for motor %d",10);
//     }

//     // RCLCPP_INFO(this->get_logger(),"camera pos: %.2f",camera_pos);

//     data_store.data[0] = camera_pos;
    
//     data_publisher->publish(data_store);
//   }
// };

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

  //Move gripper motor to appropriate location (open or closed) - handled separately from 8 motor control 
  double gripper = (int32_t)(motor_array.data[8] * (double)ENCODER_TICKS_PER_REV * (double)LEAD_SCREW_REV_PER_M) + motor_home[8];
  param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(gripper));
  param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(gripper));
  param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(gripper));
  param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(gripper));
  
  bool add_success9 = groupSyncWritePosition->addParam(9,param_goal_position);
  if (!add_success9) {
    RCLCPP_ERROR(this->get_logger(),"Failed to add motor 9");}

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

  //Set home position as current position when motor node is first initialized
  motornode->setHomePos();

  //MultiThreadedExecutor allows for processing multiple messages in parallel (necessary for camera control & data publishing with standard motor control)
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(motornode);
  executor.spin();

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
      

  return 0;
}