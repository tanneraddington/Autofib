/*=========================================================================

  Program:   Teensy UI
  Language:  C++/C

  Jason Shrand, Jesse F d'Almeida Vanderbilt University 2020. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

  =========================================================================*/

#include <cstddef>
#include <string>
#include <ros.h>                  //enable ROS communications

//ROS msg types
#include <geometry_msgs/Twist.h>  
#include <std_msgs/Float32.h>     
#include <std_msgs/Int32.h>     
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

// src files
#include "Config.h"            // Contains all constants and pin definitions
#include "src/ui_configure.h"  // Configuration code for the UI Teensy
#include "src/ForceSensor.h"   // Force sensor class 
#include "src/UITrigger.h"     // Trigger Class
#include "src/GripperWheel.h"  // Gripper Wheel Class
#include "src/LinearEncoder.h" // Linear Encoder Class
#include "src/UITouchPad.h"    // Touch

// Libraries
#include <EEPROM.h>


// This application runs a single UI controller for the prostate project.
// In the final system, two instances of this application will be executed: one for the left controller,
// and one for the right controller.

enum State_Machine { 
  UI_STATE_SETUP,   // State as soon as the UI program begins
  UI_STATE_CONFIG_MENU, // UI program is in the configuration menu
  UI_STATE_ROS_INITIALIZE, // UI program initializes ROS
  UI_STATE_NORMAL_OPERATION, // Normal operation of the UI program
  UI_STATE_FAILOVER_OPERATION // Failover mode
};

////////////////////////////////////////////////////////////////////
//                           VARIABLES                            //
////////////////////////////////////////////////////////////////////

// The state of the UI State machine
State_Machine STATE;

// The configuration settings
UIConfig::config_settings conf;

///// Variables for the ROS Publishers /////
// pointer for publisher variables
geometry_msgs::Twist ui_msg;
std_msgs::Int32 state_msg;
std_msgs::String str_msg;

ros::NodeHandle nh;                     // handler for all node stuff

//////// Sensor Objects //////
UITrigger*      uitrigger;
GripperWheel*   gripperWheel;
ForceSensor*    force_sensor;
LinearEncoder*  linearEncoder;
UITouchPad*     uitouchpad;

void clutchCallback(const std_msgs::Bool& msg){
  force_sensor->clutchCallback(msg);
}

ros::Subscriber<std_msgs::Bool>  subclutch("/user_input_clutch", &clutchCallback);

float zvel = 0.0;
float zang = 0.0;

////////////////////////////////////////////////////////////////////
//                          FUNCTIONS                             //
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////
void setup()
{
  // setup() gets called once at power-up, sets up serial debug output and Cirque's Pinnacle ASIC.

  ////////// STATE TRANSITION //////////
  STATE = UI_STATE_SETUP;
  //////////////////////////////////////

  // Attempt to establish serial connection if Teensy is connected to PC.
  // Will timeout after 5 seconds and continue without serial if no PC is connected
  Serial.begin(115200);
  waitForSerial();

  // Set pin modes
  pinMode(TOUCHPAD_LED_PIN, OUTPUT);
  pinMode(EM1_A_CH_PIN, INPUT);
  pinMode(EM1_B_CH_PIN, INPUT);
  pinMode(TRIGGER_PIN, INPUT);
  pinMode(GRIPPER_WHEEL_PIN, INPUT);

  // Serial.println("Setup");

  // Initialize touchpad
  uitouchpad = new UITouchPad(TOUCHPAD_CS_PIN, TOUCHPAD_DR_PIN, TOUCHPAD_LED_PIN);
  force_sensor = new ForceSensor(FRC_DTA, FRC_CLK);
  uitrigger = new UITrigger(TRIGGER_PIN);                         // Initialize trigger
  gripperWheel = new GripperWheel(GRIPPER_WHEEL_PIN);           // Initialize GripperWheel
  linearEncoder = new LinearEncoder(EM1_A_CH_PIN, EM1_B_CH_PIN); // initialize encoder

  uitouchpad->initialize();
  force_sensor->initSensor();
  ////////// STATE TRANSITION //////////
  STATE = UI_STATE_CONFIG_MENU;
  //////////////////////////////////////
}

////////////////////////////////////////////////////////////////////
void initializeROS()
{
  // This function initializes the ROS publisher based on the UI Configuration

  bool is_error = false;

  // try to initialize each object that has a publisher if applicable
  // catch errors if the controller conf is incorrect
  if (USE_TOUCHPAD){
    if (!uitouchpad->initPublisher(conf.control_device)) is_error = true;
  }

  if (USE_FORCE_SENSOR) force_sensor->initPublisher();

  if (USE_TRIGGER){
    if (!uitrigger->initPublisher(conf.control_device)) is_error = true;
  }
  
  if (is_error){
    // Not expected to happen, since in this state config should already have been set to valid.
    // Serial.println("Cannot initialize ROS for unknown configuration!");
  }
  
  if (!is_error)
  {
    // Serial.println("Initializing ROS Node");
    // Initialize ROS node
    nh.getHardware()->setBaud(115200);
    nh.initNode();
    
    // advertise publishers
    if (USE_TOUCHPAD) nh.advertise(*uitouchpad->getPublisher());
    if (USE_TRIGGER) nh.advertise(*uitrigger->getPublisher());
    if (USE_FORCE_SENSOR) nh.advertise(*force_sensor->getForcePublisher());

    // subscriber
    nh.subscribe(subclutch);

    ////////// STATE TRANSITION //////////
    STATE = UI_STATE_NORMAL_OPERATION;
    //////////////////////////////////////
  }
  else
  {
    // Serial.println("Transitioning to failover state");
    ////////// STATE TRANSITION //////////
    STATE = UI_STATE_FAILOVER_OPERATION;
    //////////////////////////////////////
  }
}

void waitForSerial()
{
  // Give serial time to initialize for USB
  unsigned long serial_wait_start = millis();
  unsigned long duration = 0;
  while (!Serial)
  {
    duration = millis() - serial_wait_start;
    if (duration > 5000) // If 5 seconds have passed without serial connection, then continue.
    {
      break;
    }
  }
}


////////////////////////////////////////////////////////////////////
void loop()
{
  // loop() will execute the logic based on the current state of the UI software state machine.
  //Serial.println("TEST");
  switch(STATE){
    case UI_STATE_SETUP:
      break;
    case UI_STATE_CONFIG_MENU:{
      // Check the current configuration, and give the user the option to change.
      conf = UIConfig::readConfiguration();
      String config_str = UIConfig::makeReadable(conf);

      // Serial.println("UI Configuration");
      // Serial.println(config_str);

      UIConfig::showMenu(5);

      // Load the settings again after the user has finished interaction with the config menu, if they chose to do so
      conf = UIConfig::readConfiguration();

      // switch to next valid state
      if (conf.is_valid){
        STATE = UI_STATE_ROS_INITIALIZE;
      } else{
        // Serial.println("INVALID CONFIGURATION");
        // Serial.println("Transitioning to failover state");
        ////////// STATE TRANSITION //////////
        STATE = UI_STATE_FAILOVER_OPERATION;
        //////////////////////////////////////
      }
      break;
    }
  
  case UI_STATE_ROS_INITIALIZE:
    // Initialize the ROS node based on the configuration
    initializeROS();

    ////////// STATE TRANSITION //////////
    STATE = UI_STATE_NORMAL_OPERATION;
    //////////////////////////////////////
    break;

  case UI_STATE_NORMAL_OPERATION:

    // get linear z vel from whichever sensor is defined
    if (USE_FORCE_SENSOR){
      if (force_sensor->hasNewData()){
        zvel = force_sensor->processForceSensor();
        force_sensor->publishForce();
      }

    } else if (USE_LINEAR_ENCODER){
      linearEncoder->processLinearEncoder();

      if (linearEncoder->hasNewData()){
        zvel = linearEncoder->getEncoderData();
      }
    }
    // get angular z vel from gripper wheel if defined
    if (USE_GRIPPER_WHEEL){
      gripperWheel->processGripperRotationWheel();

      if (gripperWheel->hasNewData()){
        zang = gripperWheel->getGripperData();
      }
    }

    // if touchpad defined, assemble twist information and publish
    if (USE_TOUCHPAD){
      uitouchpad->processTouchpad();

      // if(uitouchpad->hasTouchData()){
         uitouchpad->publishUITouchPad(zvel, zang);
         
      // }
    }



    // publish trigger information if defined
    if (USE_TRIGGER){
      uitrigger->processTrigger();

      if (uitrigger->hasTriggerData()){
        uitrigger->publishTrigger();
      }
    }

    nh.spinOnce();
    delay(5);
    break;

  case UI_STATE_FAILOVER_OPERATION:
    // TODO: Send ROS message notifying failure
    // Serial.println("UI is in failover mode!");
    break;
  
  }

  
    
} 
