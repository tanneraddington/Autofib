/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jesse Francisco d'Almeida Vanderbilt University 2021. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/
#ifndef FORCESENSOR_H
#define FORCESENSOR_H

#include "Arduino.h"
#include <EEPROM.h>
#include <HX711.h>
#include <ros.h> //enable ROS communications
#include <std_msgs/Float32.h> // Ros msg type
#include <std_msgs/Bool.h> // Ros msg type
#include <string>
#include <cstdlib>

class ForceSensor {
private:
    // initialize force sensor
    uint8_t datapin;
    uint8_t clkpin;
    HX711 sensor;

    // force sensor parameters
    float CALIBRATION_FACTOR = -531000.0; // calibrated to kg
    const float FORCE_SCALE_FACTOR = 1.0; // scale factor 
    // const float SPRING_CONST = .1220;     // sprint constant in N/mm
    const float SPRING_CONST = .3440;
    const float G = 9.81;                 // gravity constant 

    float minForce = -0.18;                 // min range of force sensor (kgf) TO BE CALIBRATED 
    float maxForce = 0.22;                 // max range of force sensor (kgf) new touchpad

    float minZVel = -1.0;                  // min range of z velocity (mm/s)
    float maxZVel = 1.0;                 // max range of z velocity (mm/s)
    
    float forceThreshold = 0.1;

    float force = 0.0;
    float zvel = 0.0;

    // low pass filter
    float time_const = .05;
    float lastvel = 0.0;
    float lasttime = 0.0;


    // clutch parameters
    const float clutchThreshold = 0.0003;
    bool isClutchEngaged = false; // 'engaged' defined as a full press and release such as if one were to click between modes
    bool isClutchPressed = false;     // current value of pedal from callback (true is pressed, false is unpressed)
    float stoppedVal = 0.0;
    float zeroVel = 0.0;

    enum ClutchState {
      ENGAGED, 
      DISENGAGED, 
      REENGAGED
    };
    ClutchState clutchState = ENGAGED; 

    // ROS vars
    ros::Publisher* force_msg_publisher;
    std_msgs::Float32 force_msg;

    // 

public:

    enum ForceAlg {
      LINEAR_POS,
      LINEAR_POS_CLUTCH,
      LINEAR_VEL,
      VEL_CLUTCH_REVERSE,
      VEL_CLUTCH_VARIABLE_REVERSE
    };

    ForceAlg alg;
    
    // constructor
    ForceSensor(uint8_t datapin, uint8_t clkpin);


    // initializes sensor and calibrates
    void initSensor();

    void initPublisher();

    // re-tares and recalibrates sensor
    void resetSensor();

    ////// Calculate Z Twist //////
    float processForceSensor(); // calculates the z twist to send based on desired algorithm

    float lowPassFilter(float input);
    
    ////// -------- Helpers Publishers Getters Setters -------- //////
    //// Helpers ////
    bool hasNewData();
    float getCurrForce();         // read force sensor 
    void updateTipPosition(float force);
    float calcPosPID(float goal);
    float force2dist(float force);  // converts force reading to a dist based on spring constant
    
    //// Subscribers ////
    void clutchCallback(const std_msgs::Bool& msg);

    //// Publishers ////

    // reads force sensor and publishes to ROS
    void publishForce();


    //// Getters /////
    ros::Publisher* getForcePublisher(); // return pointer to force publisher
    float getCurrPos();
    int getClutchState();
    int getLastState();
    float getStoppedVal();
    float getSpringPos();

    //// Setters ////
    void setCalibrationFactor(float calibration_factor);
    void setForceAlg(ForceAlg alg);


};


#endif // UI_CONFIGURE_H
