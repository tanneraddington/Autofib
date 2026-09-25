/*
  Program:   Teensy UI
  Language:  C++/C

  Jason Shrand Vanderbilt University 2020. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/

/**
* This file defines a set of state values to be used by the
* Teensy UI State machine
*/
#ifndef UI_STATE_H
#define UI_STATE_H

const uint8_t UI_STATE_SETUP = 0;   // State as soon as the UI program begins
const uint8_t UI_STATE_CONFIG_MENU = 1; // UI program is in the configuration menu
const uint8_t UI_STATE_ROS_INITIALIZE = 2; // UI program initializes ROS
const uint8_t UI_STATE_NORMAL_OPERATION = 3; // Normal operation of the UI program
const uint8_t UI_STATE_FAILOVER_OPERATION = 4; // Failover mode

#endif // UI_STATE_H