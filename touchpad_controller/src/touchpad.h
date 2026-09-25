/*=========================================================================

  Program:   Teensy UI
  Language:  C++/C

  Dominick Ropella, Jason Shrand, Vanderbilt University 2020. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/

/**
* This class provides functions for interacting with a single Cirque Pinnacle TM0xx0XX touchpad.

*/
#ifndef TOUCHPAD_H
#define TOUCHPAD_H

#include "Arduino.h"
#include <SPI.h>

class Touchpad
{
public:

	// Struct for retrieving touch information
	typedef struct _touchinfo
	{
	    bool is_valid;          // Whether there is actual touch data from the touchpad
		double x_pos;           // The x position of the touch, in <TBD UNITS>
		double y_pos;           // The y position of the touch, in <TBD UNITS>
		double x_disp;          // The x displacement from the previous touch, in <TBD UNITS>
		double y_disp;          // The y displacement from the previous touch, in <TBD UNITS>
        bool is_hovering;       // Whether the touch is hovering
        double touch_yDelta;    // change in x position
        double touch_xDelta;    // change in y position
        bool initial_touch;     // whether this is the first time touching
        bool has_new_touch_data;// whether there is new data to publish
	} touchinfo_t;


	/*
	* Creates a new Touchpad interface
	*
	* @param CS_pin: The chip select pin number for the touchpad sensor
	* @param DR_pin: The data ready pin number
	* @param LED_pin: The pin number for the LED
	*
	* @param use_curved_overlay: Whether the sensor uses a curved overlay or not.
	*        true  = curved overlay
	*        false = flat overlay
	*/
    Touchpad(uint8_t CS_pin, uint8_t DR_pin, uint8_t LED_pin, bool use_curved_overlay);


    /**
    * Initializes the touchpad sensor
    */
    void initialize();

    /**
    * Reads the latest touch information
    *
    * @param touchInfo: A pointer to a touchinfo_t struct to populate with data
    */
    void readTouchData(touchinfo_t &touchInfo);

    /**
     * Calculates delta position to publish to ROS 
     * @param touchInfo: A pointer to a touchinfo_t struct to populate with data
    */
    void processTouchpad(touchinfo_t &touchInfo);

    /**
    * Return a string representatoin of a touchinfo_t struct
    *
    * @param touchInfo: The touchinfo_t struct to get the string representation of
    * @param str: A string to populate with the 
    */
    static void touchDataToString(const touchinfo_t touchInfo, String &str);

private:

    /////////////////////// CONSTANTS /////////////////////

	///// Coordinate scaling values /////
	static constexpr int PINNACLE_XMAX    = 2047;    // max value Pinnacle can report for X (0 to (8 * 256) - 1)
	static constexpr int PINNACLE_YMAX    = 1535;    // max value Pinnacle can report for Y (0 to (6 * 256) - 1)
	static constexpr int PINNACLE_X_LOWER = 12;      // min "reachable" X value
	static constexpr int PINNACLE_X_UPPER = 1919;    // max "reachable" X value
	static constexpr int PINNACLE_Y_LOWER = 63;      // min "reachable" Y value
	static constexpr int PINNACLE_Y_UPPER = 1471;    // max "reachable" Y value
	static constexpr int PINNACLE_X_RANGE = (PINNACLE_X_UPPER-PINNACLE_X_LOWER);
	static constexpr int PINNACLE_Y_RANGE = (PINNACLE_Y_UPPER-PINNACLE_Y_LOWER);
	static constexpr int ZONESCALE        = 256;   // divisor for reducing x,y values to an array index for the LUT
	static constexpr int ROWS_Y           = ((PINNACLE_YMAX + 1) / ZONESCALE);
	static constexpr int COLS_X           = ((PINNACLE_XMAX + 1) / ZONESCALE);

    ///// ADC-attenuation settings (held in BIT_7 and BIT_6) /////
    // 1X = most sensitive, 4X = least sensitive
    static constexpr uint8_t ADC_ATTENUATE_1X = 0x00;
    static constexpr uint8_t ADC_ATTENUATE_2X = 0x40;
    static constexpr uint8_t ADC_ATTENUATE_3X = 0x80;
    static constexpr uint8_t ADC_ATTENUATE_4X = 0xC0;
    static constexpr uint8_t ADC_LOWER_DEFAULT = 0x02;

    // Register config values
    static constexpr uint8_t SYSCONFIG_1  = 0X00;
    static constexpr uint8_t FEEDCONFIG_1 = 0x02;
    static constexpr uint8_t FEEDCONFIG_2 = 0x1F;
    static constexpr uint8_t Z_IDLE_COUNT = 0x05;

    // Masks for Cirque Register Access Protocol (RAP)
    static constexpr uint8_t WRITE_MASK = 0x80;
    static constexpr uint8_t READ_MASK  = 0xA0;

    static constexpr uint DEFAULT_WRITE_DELAY = 50;

    // These values require tuning for optimal touch-response
    // Each element represents the Z-value below which is considered "hovering" in that XY region of the sensor.
    // The values present are not guaranteed to work for all HW configurations.
    uint8_t ZVALUE_MAP[ROWS_Y][COLS_X] =
    {
      {0, 0,  0,  0,  0,  0, 0, 0},
      {0, 2,  3,  5,  5,  3, 2, 0},
      {0, 3,  5, 15, 15,  5, 2, 0},
      {0, 3,  5, 15, 15,  5, 3, 0},
      {0, 2,  3,  5,  5,  3, 2, 0},
      {0, 0,  0,  0,  0,  0, 0, 0},
    };
    
	////////////////// STRUCT DEFINITIONS ////////////////

    // Struct to store pin values for the pad
    typedef struct _padData
    {
        uint8_t CS_Pin;
        uint8_t DR_Pin;
        uint8_t LED_Pin;
    } padData_t;

	// Convenient way to store and access measurements
	typedef struct _absData
	{
	  uint16_t xValue;
	  uint16_t yValue;
	  uint16_t zValue;
	  uint8_t buttonFlags;
	  bool touchDown;
	  bool hovering;
	} absData_t;


    ///////////////// PRIVATE MEMBERS /////////////////
    padData_t padData; // For storing pin values
    
    absData_t currentData; // For storing current touch data

    // For computing displacement from previous touches
    double x_disp = 0.0;
    double x_prev = 0.0;
    double diff_x = 0.0;
    double y_disp = 0.0;
    double y_prev = 0.0;
    double diff_y = 0.0;
    bool touch_trigger = false;


    // Whether or not the touchpad uses a curved overlay
    bool isCurvedOverlay;

    ///////////////// PRIVATE FUNCTIONS /////////////////

    ///// Pinnacle Functions //////
    void Pinnacle_Init();

    // Reads XYZ data from Pinnacle registers 0x14 through 0x17
    // Stores result in absData_t struct with xValue, yValue, and zValue members
    void Pinnacle_GetAbsolute(absData_t * result);

    // Forces Pinnacle to re-calibrate, sometimes useful when miss-compensation
    // causes touchpad to miss real touches or false touches are tringgering touch events
    void Pinnacle_forceCalibration();

    // Checks touch data to see if it is a z-idle packet (all zeros)
    static bool Pinnacle_zIdlePacket(absData_t * data);

    // Clears Status1 register flags (SW_CC and SW_DR)
    void Pinnacle_ClearFlags();

    // Enables/Disables the feed
    void Pinnacle_EnableFeed(bool feedEnable);

    // This function identifies when a finger is "hovering" so your system can choose to ignore it.
    void Pinnacle_CheckValidTouch(absData_t * touchData);

    /////  RAP Functions /////
    void RAP_Init();

    // Reads <count> Pinnacle registers starting at <address>
    void RAP_ReadBytes(byte address, byte * data, byte count);

    // Writes single-byte <data> to <address>
    void RAP_Write(byte address, byte data);

    /////  I/O Functions /////
    void Assert_CS();
    void DeAssert_CS();
    void AssertSensorLED(bool state);
    bool DR_Asserted();

    ///// ERA (Extended Register Access) Functions /////
    
    // Reads <count> bytes from an extended register at <address> (16-bit address),
    // stores values in <*data>
    void ERA_ReadBytes(uint16_t address, uint8_t * data, uint16_t count);

    // Writes a byte, <data>, to an extended register at <address> (16-bit address)
    void ERA_WriteByte(uint16_t address, uint8_t data);

    /////  Curved Overlay Functions  /////

    // Adjusts the feedback in the ADC, effectively attenuating the finger signal
    // By default, the the signal is maximally attenuated (ADC_ATTENUATE_4X for use with thin, flat overlays
    void setAdcAttenuation(uint8_t adcGain);

    // Changes thresholds to improve detection of fingers
    void tuneEdgeSensitivity();

    ///// Logical Scaling Functions /////

    // Clips raw coordinates to "reachable" window of sensor
    // NOTE: values outside this window can only appear as a result of noise
    void ClipCoordinates(absData_t * coordinates);

    // Scales data to desired X & Y resolution
    void ScaleData(absData_t * coordinates, uint16_t xResolution, uint16_t yResolution);
};

#endif // TOUCHPAD_H