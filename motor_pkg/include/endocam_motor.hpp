// Copyright 2021 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
// implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef ENDOCAM_MOTOR_NODE_HPP_ //include guards, files included once
#define ENDOCAM_MOTOR_NODE_HPP_
#include <cstdio> //import c++ libraries,input/output
#include <memory>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp" //imports ros2
#include "rcutils/cmdline_parser.h"
#include "dynamixel_sdk/dynamixel_sdk.h"
#include <std_msgs/msg/float32.hpp> //contain single floating pt (15)(30)
#include <std_msgs/msg/int32.hpp>
class EndocamMotorNode : public rclcpp::Node //name of ros2 node pub/sub/log
{
public:
 explicit EndocamMotorNode(const rclcpp::NodeOptions & options);
 void position_callback(const std_msgs::msg::Float32::SharedPtr msg);
//function called position callback that receives float 32 ros msg, code 
//in cpp
private: //only belonging to EndocamMotorNode
 rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr
motor_subscriber; //declares subscriber, when u publish this hears it
 std_msgs::msg::Float32 motor_array; //copies angle to this var
 
 double gear_ratio; //in cpp can convert end ang to motor ang to ticks
};
#endif