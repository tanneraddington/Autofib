/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jesse Francisco d'Almeida Vanderbilt University 2021. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/


#ifndef TRIGGER_H
#define TRIGGER_H

#include "Arduino.h"
#include <std_msgs/Float32.h> // Ros msg type
#include <ros.h> //enable ROS communications
#include "ui_configure.h"

class UITrigger{

private:
    uint8_t triggerPin;

    // Constant for the trigger exponential filter
    const double TRIGGER_SMOOTHING_CONSTANT  = 0.8;

    ///// Variables for the trigger /////
    double trigger_prev_reading = 0;
    double trigger_smoothing_constant = 0.8;

    // At low voltages the trigger is noisy, so you never get exactly zero when the trigger is fully pressed.
    // Consider any value of 50 or lower as "fully pressed".
    const double TRIGGER_ZERO_THRESHOLD = 1; //50

    // When trigger is partially-depressed, the signal with have small fluctuations due to variations in pressure.
    // If the value of a reading is within the jitter threshold, we will ignore it and continue with the previous reading.
    const double TRIGGER_NOISE_THRESHOLD = 2;

    double trigger_reading = 0;
    double scaled_trigger_reading = 0;

    // The min/max readings of the trigger (Determined empirically
    const double TRIGGER_CLOSED = 908;
    const double TRIGGER_OPEN = 1023;

    int new_trigger_reading;
    double filtered_reading;

    bool has_new_trigger_data = false;

    // ROS Params
    
ros::Publisher* trigger_msg_publisher;  // trigger message for opening gripper(?)
    std_msgs::Float32 trigger_msg;


public:

    UITrigger(uint8_t pin){
        this->triggerPin = pin;
    }

    /**
     * initialize the publisher based on the controller side
     * @param control_device int indicating which side device
     * @return bool on whether the control device was defined
    */
    bool initPublisher(uint8_t control_device){
        if (control_device == UIConfig::LEFT_CONTROL_DEVICE){
            this->trigger_msg_publisher = new ros::Publisher("/teensy_two/left_user_trigger_input", &this->trigger_msg);
            return true;
        } else if (control_device == UIConfig::RIGHT_CONTROL_DEVICE){
            this->trigger_msg_publisher = new ros::Publisher("/teensy_one/right_user_trigger_input", &this->trigger_msg);
            return true;
        }
        else {
            return false;
        }
    }

    /** 
     * This function reads input from the UI Trigger. filters, and marks a flag if there is a new reading
     * 
    */ 
    void processTrigger(){
        new_trigger_reading = TRIGGER_OPEN - analogRead(this->triggerPin);   // read trigger and scale 

        new_trigger_reading = max(new_trigger_reading, TRIGGER_ZERO_THRESHOLD); // set lower limit for reading

        // Check against noise threshold from previous reading
        if (abs(new_trigger_reading - trigger_reading) < TRIGGER_NOISE_THRESHOLD){
            new_trigger_reading = trigger_reading;
        }

        // Apply exponential filter
        filtered_reading = (TRIGGER_SMOOTHING_CONSTANT * trigger_reading) + (1 - TRIGGER_SMOOTHING_CONSTANT) * new_trigger_reading;

        // check if value actually changed
        if (filtered_reading != trigger_reading){
            this->has_new_trigger_data = true;
        }

        // update from previous value
        trigger_reading = filtered_reading;

        // Scale the trigger reading
        scaled_trigger_reading = trigger_reading / (TRIGGER_OPEN-TRIGGER_CLOSED); // Between 0 and 1, with 1 being fully open, and 0 being closed
    }

    /**
     * This function publishes the left_user_trigger_input or
     * right_user_trigger_input topic, depending on the configuration
     * stored in EEPROM
    */
    void publishTrigger(){
        trigger_msg.data = (float)this->scaled_trigger_reading;
        trigger_msg_publisher->publish(&this->trigger_msg);

        // reset flag
        this->has_new_trigger_data = false;
    }

    
    /////// GETTERS //////
    ros::Publisher* getPublisher(){
        return this->trigger_msg_publisher;
    }


    bool hasTriggerData(){
        return this->has_new_trigger_data;
    }
};


#endif