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

#ifndef MOTOR_NODE_HPP_
#define MOTOR_NODE_HPP_

#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <termios.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "rcutils/cmdline_parser.h"
#include "dynamixel_sdk/dynamixel_sdk.h"
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <std_msgs/msg/int32.hpp>

enum MotorState {
  MOVING,
  CAMERA_ADJUST,
  ERROR
};


class MotorNode : public rclcpp::Node
{
public:
  explicit MotorNode(const rclcpp::NodeOptions & options);
  void position_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void keyboard_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  //void setupDynamixel(uint8_t dxl_id);
  void setHomePos();
  //void moveMotors(const std::vector<float>& positions);
  void set_home_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void gripper_callback(const std_msgs::msg::Float32::SharedPtr msg);
  void camera_adjust();
  void state_machine();
  void csv_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  std::ofstream csv_file;
  bool comments; 

private:

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr motor_subscriber;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr set_home_subscriber;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gripper_sub;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr data_publisher;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr write_to_csv_subscription;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr home_publisher;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr keyboard_subscriber;
  // rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr fwkin_publisher_l;
  // rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr fwkin_publisher_r;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr fwkin_publisher_l;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr fwkin_publisher_r;
  

  //std::vector<float> motor_array;
  std_msgs::msg::Float32MultiArray motor_array;
  std_msgs::msg::Float32MultiArray data_store;
  geometry_msgs::msg::Pose fwkin_xyz_l;
  geometry_msgs::msg::Pose fwkin_xyz_r;
  int motor_array_size;
  float gripper = 0.0;
  float gripper_value;
  
  //std_msgs::msg::Float32MultiArray motor_home;
  double gear_ratio;

  //double motor_home[9];
  std::vector<int32_t> motor_home = {0,0,0,0,0,0,0,0};
  std::vector<float> offset = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  //std::vector<float> motor_home;
  double first_motor_home[9];
  double recent_trans;
  double motor_pos;
  int loops;
  int16_t goal_current = static_cast<int16_t>(1000.0 / 2.69);

  bool has_gripper = false; //flag for detecting 9th motor
  bool has_camera = false; //flag for detecting 10th motor

  double tolerance = 0.0001;
  std::vector<float> last_received_position;
  bool is_within_tolerance();
  MotorState current_state;

  rclcpp::TimerBase::SharedPtr timer_ptr;

};

#endif