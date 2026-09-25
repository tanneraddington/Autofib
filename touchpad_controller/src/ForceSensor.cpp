/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jesse Francisco d'Almeida Vanderbilt University 2021. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/

#include "ForceSensor.h"

// Constructor

/**
 * Constructor for class
 * 
 * @param datapin Data pin of the laod cell amp
 * @param clkpin  Clock pin of the load cell amp 
*/
ForceSensor::ForceSensor(uint8_t datapin, uint8_t clkpin){
    this->datapin = datapin;
    this->clkpin = clkpin;
}


////// Inits //////

/**
 *  initializes the force sensor object with the pins
 *  sets the scale with the default calibration factor and tares
 * 
*/
void ForceSensor::initSensor(){
    this->sensor.begin(this->datapin, this->clkpin);
    this->sensor.set_scale(this->CALIBRATION_FACTOR);
    this->sensor.tare();
}

/**
 *  initializes the publisher(s) with their topic name and message address
*/
void ForceSensor::initPublisher(){
    this->force_msg_publisher = new ros::Publisher("force_read", &this->force_msg);
    
}

/**
 * resets the calibration and tares the force sensor
*/
void ForceSensor::resetSensor(){
    this->sensor.set_scale(this->CALIBRATION_FACTOR);
    this->sensor.tare();
}


////// Calcualte Z Twist //////


/**
 * main function call
 * calculates the z twist based on the desired algorithm
 * @param alg string to choose the desired algorithm
 * @return z twist to be published in main
*/
float ForceSensor::processForceSensor(){
    this->force = this->getCurrForce();    // get current force
    
    // this->zvel = this->lowPassFilter(map(this->force, this->minForce, this->maxForce, this->minZVel, this->maxZVel));

    this->zvel = map(this->force, this->minForce, this->maxForce, this->minZVel, this->maxZVel);
    if (this->zvel < this->minZVel){
        this->zvel = this->minZVel;
    }
    else if(this->zvel > this->maxZVel){
        this->zvel = this->maxZVel;
    }

    this->lastvel = this->zvel;
    
    // return constrain(this->zvel, 0.0, 1.0);
    return this->zvel;
}


float ForceSensor::lowPassFilter(float x){
    unsigned long timestamp = micros();

    float dt = (timestamp - this->lasttime) *1e-6f;

    // quick fix for strange cases (micros overflow)
      if (dt < 0.0f || dt > 0.5f) dt = 1e-3f;

      // calculate the filtering 
      float alpha = this->time_const/(this->time_const + dt);
      float y = alpha*this->lastvel + (1.0f - alpha)*x;
      // save the variables
      this->lasttime = timestamp;
      return y;
}

////// -------- Helpers Publishers Getters Setters -------- //////

//// Helpers ////

bool ForceSensor::hasNewData(){
    return this->sensor.is_ready();
}

/**
 * reads force sensor and caps it at a max
 * 
 * @return reading of the force sensor
*/
float ForceSensor::getCurrForce(){
    return this->sensor.get_units() * this->FORCE_SCALE_FACTOR;

}



//// Subscribers /////

/**
 * callback for the clutch pedal
 *  switches state of clutch engagement when pedal is released
 * TODO: add in debouncing or deadband region as needed
 * 
 * @param msg bool pedal message 
*/
void ForceSensor::clutchCallback(const std_msgs::Bool& msg){
    // switch clutch engagement when button is released (ie last state is 1 and curr is 0)
    if (isClutchPressed && !msg.data){
        this->isClutchEngaged = !this->isClutchEngaged;
    }
    
    // update clutch state with current reading
    this->isClutchPressed = msg.data;
}

//// Publishers /////

/**
 * retrieves current force reading and publishes it to the force topic
*/
void ForceSensor::publishForce(){
    this->force_msg.data = this->getCurrForce();
    this->force_msg_publisher->publish(&this->force_msg);
}


///// Getters /////

/**
 * returns the publisher object
 * 
 * @return force msg publisher
 */
ros::Publisher* ForceSensor::getForcePublisher(){
    return this->force_msg_publisher;
}


int ForceSensor::getClutchState(){
    return this->clutchState;
}

int ForceSensor::getLastState(){
    return this->clutchState;
}

float ForceSensor::getStoppedVal(){
    return this->stoppedVal;
}


///// Setters /////

/**
 *  sets a new calibration factor
 *  DOES NOT reset the sensor, must run resetSensor() after
 * 
 *  @param calibration_factor new calibration factor
*/
void ForceSensor::setCalibrationFactor(float calibration_factor){
    this->CALIBRATION_FACTOR = CALIBRATION_FACTOR;
}

void ForceSensor::setForceAlg(ForceAlg alg){
    this->alg = alg;
}


