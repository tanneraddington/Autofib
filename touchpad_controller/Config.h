/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jesse Francisco d'Almeida Vanderbilt University 2021. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/

// DEFINES constants and hardware pins for main code
#ifndef CONFIG_H
#define CONFIG_H

//// SENSOR FLAGS ////
// defines which sensors are in use
#define USE_FORCE_SENSOR 1
#define USE_TOUCHPAD 1
#define USE_TRIGGER 0
#define USE_GRIPPER_WHEEL 0
#define USE_LINEAR_ENCODER 0

//// DEBUG FLAGS ////
// some might be depricated
// 0: Debug output off    1: Debug output on
#define DEBUG_TOUCHPAD 0
#define DEBUG_TRIGGER 0
#define DEBUG_GRIPPER_WHEEL 0
#define DEBUG_LINEAR_ENCODER 0
#define DEBUG_ROS 0
#define DEBUG_FORCE_SENSOR 0 


// Hardware pin-number labels /////

// Pins for the touchpad
#define TOUCHPAD_CS_PIN 10    // Chip Select for touchpad sensor
#define TOUCHPAD_DR_PIN 9       // Data-Ready for touchpad sensor
#define TOUCHPAD_LED_PIN 13     // LED for touchpad sensor

// Pins for the gripper rotation wheel
#define GRIPPER_WHEEL_PIN 14 // TODO - Replace with real pin

// Pins for the EM1 linear encoder 
//ATTENTION: the teensies comunicates over conn7 or conn6 depending on the side
//Conn6 - Left handle
#define EM1_A_CH_PIN 23
#define EM1_B_CH_PIN 22

// Pins for force sensor data and clock
#define FRC_DTA  A5
#define FRC_CLK  A4

//Conn7 - Right handle
//#define EM1_A_CH_PIN 21
//#define EM1_B_CH_PIN 20

// Pins for the trigger
#define TRIGGER_PIN 15 // Conn4 IO




#endif
