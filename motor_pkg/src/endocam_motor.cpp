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

/************************************************************************
// This example is written for DYNAMIXEL X (excluding XL-320) and MX (2.0)
// series with U2D2.
// For other series, please refer to the product eManual and modify the
// Control Table addresses and other definitions.
//
// Author: Will Son
// Note: this example does not apply with our now edited code
************************************************************************/

// Takes in array giving position and rotation commands and sends motors
// to those positions.
// Can change motor operating mode in setupDynamixel().

#include <cstdio>
#include <memory>
#include <string>
#include <cstdint>
#include <cmath>

#include <dynamixel_sdk/dynamixel_sdk.h>
#include <rclcpp/rclcpp.hpp>

#include "rcutils/cmdline_parser.h"
#include "dynamixel_node.hpp"
#include "endocam_motor.hpp"

#include <std_msgs/msg/float32.hpp>


// Control table address for X series (except XL-320)
#define ADDR_OPERATING_MODE 11
#define ADDR_TORQUE_ENABLE 64
#define ADDR_GOAL_POSITION 116
#define ADDR_GOAL_CURRENT 102
#define ADDR_PRESENT_POSITION 132

// Protocol version
#define PROTOCOL_VERSION 2.0

// Default setting
// #define BAUDRATE 115200
#define BAUDRATE 57600

#define DEVICE_NAME "/dev/ttyUSB0"

#define ENCODER_TICKS_PER_REV 4096


#define MOTOR_ID 10


// Angle limits
const float HOME_ANGLE = 15.0f;
const float MIN_ANGLE = 17.0f;
const float MAX_ANGLE = 88.0f;

const float TICKS_PER_DEGREE =
  ENCODER_TICKS_PER_REV / 360.0f;


// Global variables
int32_t home_position_ticks = 0;

dynamixel::PortHandler * portHandler = nullptr;
dynamixel::PacketHandler * packetHandler = nullptr;

uint8_t dxl_error = 0;
int dxl_comm_result = COMM_TX_FAIL;

dynamixel::GroupSyncWrite * groupSyncWritePosition = nullptr;
dynamixel::GroupSyncRead * groupSyncRead = nullptr;


// Constructor
EndocamMotorNode::EndocamMotorNode(
  const rclcpp::NodeOptions & options)
: Node("endocam_motor_node", options)
{
  RCLCPP_INFO(
    this->get_logger(),
    "Initializing motor node");

  // Dynamixel functions to write encoder positions to motors
  groupSyncWritePosition =
    new dynamixel::GroupSyncWrite(
      portHandler,
      packetHandler,
      ADDR_GOAL_POSITION,
      4);

  groupSyncRead =
    new dynamixel::GroupSyncRead(
      portHandler,
      packetHandler,
      ADDR_PRESENT_POSITION,
      4);


  // Listen for motor angle commands
  motor_subscriber =
    this->create_subscription<std_msgs::msg::Float32>(
      "motor_angle",
      10,
      std::bind(
        &EndocamMotorNode::position_callback,
        this,
        std::placeholders::_1));


  // Establish gear ratio
  if (this->has_parameter("gear_ratio"))
  {
    gear_ratio =
      this->get_parameter("gear_ratio").as_double();
  }
  else
  {
    this->declare_parameter("gear_ratio", 1.0);

    gear_ratio =
      this->get_parameter("gear_ratio").as_double();
  }


  // Initialize motor array
  motor_array.data = 0.0f;
}


// Callback function to control motor position
void EndocamMotorNode::position_callback(
  const std_msgs::msg::Float32::SharedPtr msg)
{
  // Clear previous commands
  groupSyncWritePosition->clearParam();

  // Store requested angle
  motor_array.data = msg->data;

  // Goal position must be an integer tick count
  float angle = motor_array.data;


  // Safety limits
  if (angle < MIN_ANGLE)
  {
    RCLCPP_WARN(
      this->get_logger(),
      "Input angle is outside of range. Moving to %.1f degrees.",
      MIN_ANGLE);
    
    angle = MIN_ANGLE;
  }

  if (angle > MAX_ANGLE)
  {
    RCLCPP_WARN(
      this->get_logger(),
      "Input angle is outside of range. Moving to %.1f degrees.",
      MAX_ANGLE);
    angle = MAX_ANGLE;
  }


  // Difference from home angle
  float offset_angle =
    angle - HOME_ANGLE;


  // Convert angle to encoder ticks
  int32_t goal_position =
    home_position_ticks +
    static_cast<int32_t>(
      offset_angle * TICKS_PER_DEGREE);


  // Convert goal position to four-byte Dynamixel parameter
  uint8_t param_goal_position[4];

  param_goal_position[0] =
    DXL_LOBYTE(DXL_LOWORD(goal_position));

  param_goal_position[1] =
    DXL_HIBYTE(DXL_LOWORD(goal_position));

  param_goal_position[2] =
    DXL_LOBYTE(DXL_HIWORD(goal_position));

  param_goal_position[3] =
    DXL_HIBYTE(DXL_HIWORD(goal_position));


  // Add motor command to packet
  bool add_success =
    groupSyncWritePosition->addParam(
      MOTOR_ID,
      param_goal_position);


  if (!add_success)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Failed to add rotation command.");

    return;
  }


  // Send command
  dxl_comm_result =
    groupSyncWritePosition->txPacket();


  if (dxl_comm_result != COMM_SUCCESS)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Failed to send position command.");
  }


  // Clear motor commands
  groupSyncWritePosition->clearParam();
}


// Motor setup function
void setupDynamixel(uint8_t dxl_id)
{
  // Disable torque
  dxl_comm_result =
    packetHandler->write1ByteTxRx(
      portHandler,
      dxl_id,
      ADDR_TORQUE_ENABLE,
      0,
      &dxl_error);


  if (dxl_comm_result != COMM_SUCCESS)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("endocam_motor_node"),
      "Failed to disable torque.");
  }
  else
  {
    RCLCPP_INFO(
      rclcpp::get_logger("endocam_motor_node"),
      "Succeeded to disable torque.");
  }


  // Set operating mode
  dxl_comm_result =
    packetHandler->write1ByteTxRx(
      portHandler,
      dxl_id,
      ADDR_OPERATING_MODE,
      4,
      &dxl_error);


  if (dxl_comm_result != COMM_SUCCESS)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("endocam_motor_node"),
      "Failed to set Position Control Mode.");
  }
  else
  {
    RCLCPP_INFO(
      rclcpp::get_logger("endocam_motor_node"),
      "Succeeded to set Position Control Mode.");
  }
}


// Main function
int main(int argc, char * argv[])
{
  // Initialize ROS 2
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;

  options.allow_undeclared_parameters(true)
    .automatically_declare_parameters_from_overrides(true);


  // Create Dynamixel communication objects
  portHandler =
    dynamixel::PortHandler::getPortHandler(
      DEVICE_NAME);

  packetHandler =
    dynamixel::PacketHandler::getPacketHandler(
      PROTOCOL_VERSION);


  // Create ROS node
  auto endocam_motor_node =
    std::make_shared<EndocamMotorNode>(options);


  // Open serial port
  dxl_comm_result =
    portHandler->openPort();


  if (dxl_comm_result == false)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("endocam_motor_node"),
      "Failed to open the port!");

    rclcpp::shutdown();

    return -1;
  }
  else
  {
    RCLCPP_INFO(
      rclcpp::get_logger("endocam_motor_node"),
      "Succeeded to open the port.");
  }


  // Set baudrate
  dxl_comm_result =
    portHandler->setBaudRate(BAUDRATE);


  if (dxl_comm_result == false)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("endocam_motor_node"),
      "Failed to set the baudrate!");

    portHandler->closePort();
    rclcpp::shutdown();

    return -1;
  }
  else
  {
    RCLCPP_INFO(
      rclcpp::get_logger("endocam_motor_node"),
      "Succeeded to set the baudrate.");
  }


  // Initialize motor
  setupDynamixel(MOTOR_ID);


  // Add motor to synchronous read
  bool add_read_success =
    groupSyncRead->addParam(MOTOR_ID);


  if (!add_read_success)
  {
    RCLCPP_ERROR(
      endocam_motor_node->get_logger(),
      "Failed to add motor ID to sync read.");

    portHandler->closePort();
    rclcpp::shutdown();

    return -1;
  }


  // Read home position
  dxl_comm_result =
    groupSyncRead->txRxPacket();


  if (dxl_comm_result != COMM_SUCCESS)
  {
    RCLCPP_ERROR(
      endocam_motor_node->get_logger(),
      "Failed to read home position.");

    portHandler->closePort();
    rclcpp::shutdown();

    return -1;
  }


  // Store home position
  home_position_ticks =
    groupSyncRead->getData(
      MOTOR_ID,
      ADDR_PRESENT_POSITION,
      4);


  RCLCPP_INFO(
    endocam_motor_node->get_logger(),
    "Home position: %d ticks",
    home_position_ticks);


  // Enable torque
  dxl_comm_result =
    packetHandler->write1ByteTxRx(
      portHandler,
      MOTOR_ID,
      ADDR_TORQUE_ENABLE,
      1,
      &dxl_error);


  if (dxl_comm_result != COMM_SUCCESS)
  {
    RCLCPP_ERROR(
      endocam_motor_node->get_logger(),
      "Failed to enable torque.");
  }
  else
  {
    RCLCPP_INFO(
      endocam_motor_node->get_logger(),
      "Succeeded to enable torque.");
  }


  // Spin ROS 2 node
  rclcpp::executors::MultiThreadedExecutor executor;

  executor.add_node(endocam_motor_node);

  executor.spin();


  // Disable torque when ROS shuts down
  packetHandler->write1ByteTxRx(
    portHandler,
    MOTOR_ID,
    ADDR_TORQUE_ENABLE,
    0,
    &dxl_error);


  // Close port
  portHandler->closePort();


  // Clean up Dynamixel objects
  delete groupSyncWritePosition;
  delete groupSyncRead;


  // Shutdown ROS
  rclcpp::shutdown();

  return 0;
}
