/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jesse Francisco d'Almeida Vanderbilt University 2021. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/

#ifndef LINEARENCODER_H
#define LINEARENCODER_H

#include "Arduino.h"
#include <ros.h> //enable ROS communications
#include <Encoder.h>              // encoder library


class LinearEncoder{

private:
    // pins
    uint8_t pin_cha;
    uint8_t pin_chb;

    // This value is used as the initial position for the linear encoder
    // before any real values have been read from the device. By checking
    // for this value, we detect the first touch, and set the initial z velocity
    // for the encoder to zero
    const int NO_ENCODER_DATA = -999;

    ///// Variables for linear encoder /////
    Encoder* encoder;
    float encoderPosition = NO_ENCODER_DATA;
    float encoderPosDelta = 0;
    float encoderScaleFactor = 0.006;
    float encoderZeroPoint = 25;
    float newValue;

    bool has_new_encoder_data = false;

public:
    LinearEncoder(uint8_t pinA, uint8_t pinB){
        this->pin_cha = pinA;
        this->pin_chb = pinB;

        this->encoder = new Encoder(pinA, pinB);
    }

    /**
     * Processes the input from the linear encoder for z-translation
    */
    void processLinearEncoder(){
        newValue  = encoder->read();
        if (encoderPosition == NO_ENCODER_DATA){
        // This is the first reading of the encoder position, so set the
        // position delta to 0.
        encoderPosDelta = 0;
        }
        else{   
        // There is a previous reading, so compute the delta as the difference
        // between the new position and the last reported position
        
            if (newValue < 40 && newValue > -30){
                encoderPosDelta = 0;
            }
            else{
            //ATTENTION - sign for left handle is positiv - for right handle negative
            encoderPosDelta = newValue*encoderScaleFactor;
            }
        }

        encoderPosition = newValue;
        has_new_encoder_data = true;
    }

    /**
     * @return bool is there is new data from the encoder
    */
    bool hasNewData(){
        return this->has_new_encoder_data;
    }

    float getEncoderData(){
        this->has_new_encoder_data = false;
        return this->encoderPosition;
    }




};

#endif