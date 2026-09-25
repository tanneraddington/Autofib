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
  void position_callback(const std_msgs::msg::Float32::SharedPtr msg);

private:

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr motor_subscriber;

  //std::vector<float> motor_array;
  std_msgs::msg::Float32 motor_array;
  int motor_array_size;
  
  //std_msgs::msg::Float32MultiArray motor_home;
  double gear_ratio;

  double motor_pos;


  std::vector<float> last_received_position;
  bool is_within_tolerance();
  MotorState current_state;

  rclcpp::TimerBase::SharedPtr timer_ptr;

};

#endif