/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jesse Francisco d'Almeida Vanderbilt University 2021. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/
#ifndef UI_TOUCHPAD_H
#define UI_TOUCHPAD_H


#include "Arduino.h"
#include "touchpad.h"      // Cirque Pinnacle Touchpad
#include "ui_configure.h"
#include <geometry_msgs/Twist.h>  

class UITouchPad{
private:
  // pin defs
  uint8_t cspin;
  uint8_t drpin;
  uint8_t ledpin;

  const int gain = 4.0;

  Touchpad* touchpad;
  Touchpad::touchinfo_t touchinfo;
  double touch_prevX;
  double touch_prevY;
  double touch_xDelta;
  double touch_yDelta;

  double touch_x;
  double touch_y;
  long lasttouchtime = 0;

  // max travel displacement in x and y for mapping
  double maxXTravel = .75;
  double minXTravel = -.75;
  double maxYTravel = maxXTravel;
  double minYTravel = minXTravel;


  int PINNACLE_X_LOWER = 98;      // min "reachable" X value
  int PINNACLE_X_UPPER = 950;    // max "reachable" X value
  int PINNACLE_Y_LOWER = 70;      // min "reachable" Y value
  int PINNACLE_Y_UPPER = 950;    // max "reachable" Y value
  int PINNACLE_X_RANGE = (PINNACLE_X_UPPER-PINNACLE_X_LOWER);
  int PINNACLE_Y_RANGE = (PINNACLE_Y_UPPER-PINNACLE_Y_LOWER);


  // Used to keep track of whether a touchpad data point is the initial
  // touch, or if it's the continuation of a finger drag across the pad.
  bool initial_touch = true;

  bool has_new_touch_data = false;
  bool has_contact = true; // keep track of if there is contant 
  float noContactValue = -100; // arbitrary impossible value to determine if there is nocontact but impossible for touchpad to actually reach during normal opertation

  // ros msgs
  ros::Publisher* ui_msg_publisher;       // twist message for tip position
  geometry_msgs::Twist ui_msg;

  // lowpass filter
  float time_const = .05;
  double lastX = 0.0;
  double lastY = 0.0;
  double lasttimeX = 0.0;
  double lasttimeY = 0.0;

public:

  UITouchPad(uint8_t cspin, uint8_t drpin, uint8_t ledpin){
    this->touchpad = new Touchpad(cspin, drpin, ledpin, true);
  }

  void initialize(){
    this->touchpad->initialize();
  }

  /**
   * initialize the publisher based on the controller side
   * @param control_device int indicating which side device
   * @return bool on whether the control device was defined
  */
  bool initPublisher(uint8_t control_device){
      if (control_device == UIConfig::LEFT_CONTROL_DEVICE){
        this->ui_msg_publisher = new ros::Publisher("/left_user_twist_input", &ui_msg);
        return true;
      } else if (control_device == UIConfig::RIGHT_CONTROL_DEVICE){
        ui_msg_publisher = new ros::Publisher("/right_user_twist_input", &ui_msg);
        return true;          
      } else {
        return false;
      }
  }
  
  ////////////////////////////////////////////////////////////////////
  void processTouchpad(){
    // This function processes input from the Pinnacle touchpad for x and y movement

    touchpad->readTouchData(touchinfo);

    // Indicate that data is ready to send over ROS if the data is valid
    // if (touchinfo.is_valid){
    //   this->has_contact = true;
    //   // if (initial_touch){
    //   //   // This is the first data point where the user is touching the touchpad, so start
    //   //   // the displacement at zero
    //   //   //touch_prevX = 0;
    //   //   //touch_prevY = 0;
    //   //   touch_prevX = touchinfo.x_disp;
    //   //   touch_prevY = touchinfo.y_disp;
    //   //   touch_xDelta = 0;
    //   //   touch_yDelta = 0;
    //   //   initial_touch = false;
    //   // } else{
    //   //   // Update xDelta and yDelta as the difference between current reading and previous
    //   //   touch_xDelta = touchinfo.x_disp - touch_prevX;
    //   //   touch_yDelta = touchinfo.y_disp - touch_prevY;

    //   //   // Update touch_prevX and touch_prevY to current reading, so the delta can be computed next time
    //   //   // touch_prevX = touchinfo.x_disp;
    //   //   // touch_prevY = touchinfo.y_disp;
    //   // }
    //   has_new_touch_data = true;
    //   scaleData(&touchinfo);
    //   touch_x = touchinfo.x_pos;
    //   touch_y = touchinfo.y_pos;

      

    //   lasttouchtime = millis();

    // } else{
    //   // The user is not touching the pad
    //   if ((millis() - lasttouchtime) > 100){
    //     this->has_contact = false;
    //     // initial_touch = true; // Reset so the next time the user touches the pad, the velocity will start at zero
    //     this->has_new_touch_data = false;
    //     // touch_prevX = 0;
    //     // touch_prevY = 0;
    //     // touch_xDelta = 0;
    //     // touch_yDelta = 0;
    //     touch_x = 0;
    //     touch_y = 0;
    //   }
    // }


    // Simplified
    // Indicate that data is ready to send over ROS if the data is valid
    if (touchinfo.is_valid){
      this->has_contact = true;
      has_new_touch_data = true;
      scaleData(&touchinfo);
      touch_x = touchinfo.x_pos;
      touch_y = touchinfo.y_pos;

      lasttouchtime = millis();
    } else{ // The user is not touching the pad

        this->has_contact = false;
        this->has_new_touch_data = false;
        touch_x = 0;
        touch_y = 0;
    }

    // Serial.printf("X: %f  Y: %f  valid: %d  has contact: %d \n", touch_x, touch_y, touchinfo.is_valid, this->has_contact);
    // Serial.println("-------");
  }


  /**
   * assemble the complete twist message from other sources and publish
   * @param zvel linear velocity in the z (either from force sensor or encoder)
   * @param zang angular velocity in the z (from gripper wheel if applicable)
  */
  void publishUITouchPad(float zvel, float zang){

    // if no contact, dont change the z velocity
    // if(!this->has_contact){
    //     zvel = 0;
    //     touch_x = 0;
    //     touch_y = 0;
    // }

    touch_x = map(touch_x, this->minXTravel, this->maxXTravel, -1.0, 1.0);
    touch_y = map(touch_y, this->minYTravel, this->maxYTravel, -1.0, 1.0);
    

    // this->ui_msg.linear.y = -this->lowPassFilterX(constrain(touch_x, -1.0, 1.0));
    // this->ui_msg.linear.x = this->lowPassFilterY(constrain(touch_y, -1.0, 1.0));

    this->ui_msg.linear.y = -constrain(touch_x, -1.0, 1.0);
    this->ui_msg.linear.x = constrain(touch_y, -1.0, 1.0);

    this->ui_msg.linear.z = zvel;

    if(!this->has_contact){
      this->ui_msg.angular.x = this->noContactValue;
    }
    else{
      this->ui_msg.angular.x = 0;
    }
    
    this->ui_msg.angular.z = zang;


    // this->ui_msg.linear.x = constrain(this->ui_msg.linear.x, -1.0, 1.0);
    // this->ui_msg.linear.y = constrain(this->ui_msg.linear.y, -1.0, 1.0);

    // Serial.printf("X: %.2f  Y: %.2f  Z: %.2f   Radius %.3f \n", this->ui_msg.linear.x, this->ui_msg.linear.y, this->ui_msg.linear.z, sqrt(touch_x*touch_x + touch_y*touch_y));
    // Serial.println(x, y, z);
    Serial.println("%.2f, %.2f, %.2f, %.2f", ui_msg.linear.x, ui_msg.linear.y, ui_msg.linear.z, ui_msg.angular.x);
    
    this->ui_msg_publisher->publish(&this->ui_msg);
    this->has_new_touch_data = false;
  }

  /////// GETTERS //////
  ros::Publisher* getPublisher(){
      return this->ui_msg_publisher;
  }


  bool hasTouchData(){
      return this->has_new_touch_data;
  }

  void scaleData(Touchpad::touchinfo_t * coordinates){
    
    double xTemp = constrain(coordinates->x_pos, PINNACLE_X_LOWER, PINNACLE_X_UPPER);
    double yTemp = constrain(coordinates->y_pos, PINNACLE_Y_LOWER, PINNACLE_Y_UPPER);

    // translate coordinates to (0, 0) reference by subtracting edge-offset
    // xTemp -= PINNACLE_X_LOWER;
    // yTemp -= PINNACLE_Y_LOWER;

    // scale coordinates to (xResolution, yResolution) range
    coordinates->x_pos = map(xTemp, PINNACLE_X_LOWER, PINNACLE_X_UPPER, -1.0, 1.0);
    coordinates->y_pos = map(yTemp, PINNACLE_Y_LOWER, PINNACLE_Y_UPPER, -1.0, 1.0);

  }
  
  float lowPassFilterX(float inVal){
    unsigned long timestamp = micros();

    float dt = (timestamp - this->lasttimeX) *1e-6f;

    // quick fix for strange cases (micros overflow)
    if (dt < 0.0f || dt > 0.5f) dt = 1e-3f;

    // calculate the filtering 
    float alpha = this->time_const/(this->time_const + dt);
    float outVal = alpha*this->lastX + (1.0f - alpha)*inVal;
    // save the variables
    this->lasttimeX = timestamp;
    this->lastX = outVal;
    return outVal;
  }

  float lowPassFilterY(float inVal){
    unsigned long timestamp = micros();

    float dt = (timestamp - this->lasttimeY) *1e-6f;

    // quick fix for strange cases (micros overflow)
    if (dt < 0.0f || dt > 0.5f) dt = 1e-3f;

    // calculate the filtering 
    float alpha = this->time_const/(this->time_const + dt);
    float outVal = alpha*this->lastY + (1.0f - alpha)*inVal;
    // save the variables
    this->lasttimeY = timestamp;
    this->lastY = outVal;
    return outVal;
  }

};

#endif

