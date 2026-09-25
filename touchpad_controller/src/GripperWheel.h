/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jesse Francisco d'Almeida Vanderbilt University 2021. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/

#ifndef GRIPPERWHEEL_H
#define GRIPPERWHEEL_H


#include <cstddef>
#include "Arduino.h"
#include <ros.h> //enable ROS communications
#include "AverageFilter.h" // Moving average filter for the gripper wheel


class GripperWheel{

private:

    uint8_t pin;

    const double GRIPPER_WHEEL_SCALE_FACTOR = 1;

    const int GRIPPER_WHEEL_MIN = 1; // The minimum value reported by the wheel (Rotated all the way left)
    const int GRIPPER_WHEEL_MAX = 950; // The maximum value reported by the wheel (Rotated all the way right)
    const int GRIPPER_WHEEL_RANGE = GRIPPER_WHEEL_MAX - GRIPPER_WHEEL_MIN;

    // When the gripper wheel is tuned to any setting, there will be an amount of noise in the reading.
    // If any change is below the noise threshold, then it will not be interpreted as an actual change in
    // the gripper rotation setting. Note that the noise is highest when the wheel is set halfway.
    // This is the amount of noise that we should tolerate.
    const int GRIPPER_WHEEL_NOISE_THRESHOLD = 25;

    // constant for the gripper wheel exponential filter
    const double GRIPPER_WHEEL_SMOOTHING_CONSTANT = 0.6; //0.8 initially set

    // Constant for the gripper moving average filter size
    const std::size_t GRIPPER_WHEEL_AVERAGE_FILTER_SIZE = 10;
    // const uint8_t GRIPPER_WHEEL_AVERAGE_FILTER_SIZE = 10;


    // Keep track of the previous reading (in encoder counts) to apply an exponential filter for the gripper wheel
    double gripper_wheel_prev_reading = 0;
    double gripper_wheel_delta = 0;
    int gripper_avg_filter_size = 10;
    AverageFilter* gripper_filter;

    double clamped_gripper_wheel_reading = 0; // The clamped value for the gripper wheel. -1 is all the way to the left, 1 is all the way to the right, and 0 is centered.

    // Used to keep track of whether a touchpad data point is the initial
    // touch, or if it's the continuation of a finger drag across the pad.
    //bool initial_touch = true;

    ///// Variables for the gripper rotation wheel //////
    //const int GRIPPER_WHEEL_MIN = 1; // The minimum value reported by the wheel (Rotated all the way left)
    //const int GRIPPER_WHEEL_MAX = 950; // The maximum value reported by the wheel (Rotated all the way right)

    // When the gripper wheel is tuned to any setting, there will be an amount of noise in the reading.
    // If any change is below the noise threshold, then it will not be interpreted as an actual change in
    // the gripper rotation setting. Note that the noise is highest when the wheel is set halfway.
    // This is the amount of noise that we should tolerate.
    const int GRIPPER_WHEEL_NOISE_TOLERANCE = 30;

    // It was empirically shown that if there is physical movement to the UI Controller, then there can be large.
    // brief spikes in the gripper wheel reading that are above the noise threshold. To counter this, the UI will
    // use a moving average of the


    bool has_new_gripper_wheel_data = false;
    int wheel_reading;
    int avg_reading;
    double filtered_reading;
    double prev_clamped_reading;

public:
    GripperWheel(uint8_t pin){
      this->pin = pin;
      this->gripper_filter = new AverageFilter(GRIPPER_WHEEL_AVERAGE_FILTER_SIZE);
    }


    /**
     * reads input from the gripper wheel
     * creates flag if there is new data
    */
    void processGripperRotationWheel(){

      wheel_reading = analogRead(this->pin);

      //// First, use moving average filter as a first pass to reduce noise to
      // Add reading to the moving average filter, and get the new average
      gripper_filter->pushValue((double)wheel_reading);
      avg_reading = round(gripper_filter->getAverage());

      // if (DEBUG_GRIPPER_WHEEL)
      // {
      //   Serial.print("Gripper Wheel (Unsmoothed): "); Serial.print(wheel_reading); Serial.print(", ");
      //   Serial.print("Moving avg: "); Serial.print(avg_reading); Serial.print(", ");
      // }

      // Check filtered reading against allowed noise threshold to determine if we've moved or not.
      if (abs(avg_reading - gripper_wheel_prev_reading) < GRIPPER_WHEEL_NOISE_THRESHOLD){
        avg_reading = gripper_wheel_prev_reading;
      }

      // Next, apply exponential filter so we get a smooth transition between changes above the noise threshold
      filtered_reading = (GRIPPER_WHEEL_SMOOTHING_CONSTANT * gripper_wheel_prev_reading) + (1 -  GRIPPER_WHEEL_SMOOTHING_CONSTANT) * avg_reading;

      // If the filtered reading is within the noise threshold from either the max or the min, then "snap" the value to the limit
      if (filtered_reading - GRIPPER_WHEEL_MIN <= GRIPPER_WHEEL_NOISE_THRESHOLD){
        filtered_reading = GRIPPER_WHEEL_MIN;
      }
      else if (GRIPPER_WHEEL_MAX - filtered_reading <= GRIPPER_WHEEL_NOISE_THRESHOLD){
        filtered_reading = GRIPPER_WHEEL_MAX;
      }


      if (filtered_reading != gripper_wheel_prev_reading){
        // Compute the clamped value, between -1 and 1
        clamped_gripper_wheel_reading = (filtered_reading + GRIPPER_WHEEL_MIN) / GRIPPER_WHEEL_RANGE; // Clamped between 0 and 1
        clamped_gripper_wheel_reading = (clamped_gripper_wheel_reading * 2) - 1; // Clamped between -1 and 1

        // Compute the delta from the previous reading
        prev_clamped_reading = (gripper_wheel_prev_reading + GRIPPER_WHEEL_MIN) / GRIPPER_WHEEL_RANGE;
        prev_clamped_reading = (prev_clamped_reading * 2) - 1; // Clamped between -1 and 1

        //gripper_wheel_delta = clamped_gripper_wheel_reading - prev_clamped_reading;
        gripper_wheel_delta = clamped_gripper_wheel_reading;

        this->has_new_gripper_wheel_data = true;
      }

      gripper_wheel_prev_reading = filtered_reading;

      // if (DEBUG_GRIPPER_WHEEL)
      // {
      //   Serial.print("Gripper Wheel (Smoothed): "); Serial.print(filtered_reading); Serial.println();
      // }
    }



    /**
     * @return bool is there is new data from the wheel
    */
    bool hasNewData(){
      return this->has_new_gripper_wheel_data;
    }

    /**
     * returns data and resets flag
     * @return current gripper data
    */
    float getGripperData(){
      this->has_new_gripper_wheel_data = false;
      return this->gripper_wheel_delta * GRIPPER_WHEEL_SCALE_FACTOR;
    }







};



#endif