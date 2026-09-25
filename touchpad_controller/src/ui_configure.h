/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jason Shrand Vanderbilt University 2020. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/
#ifndef UI_CONFIGURE_H
#define UI_CONFIGURE_H


#include "Arduino.h"
#include <EEPROM.h>

namespace UIConfig {

////////////////////// CONSTANTS ////////////////////////

// EEPROM AddresseS storing UI configuration
const uint8_t CONTROL_DEVICE_EEPROM_ADDR = 0;

// UI Configuration enum variables
const uint8_t UNDEFINED_CONTROL_DEVICE = 0;
const uint8_t LEFT_CONTROL_DEVICE = 1;
const uint8_t RIGHT_CONTROL_DEVICE = 2;


/////////////////// CONFIGURATION STRUCT //////////////////
// This struct is returned by readConfiguration(), and 
// contains the configuration data for the UI controller
struct config_settings {
    bool is_valid; // Whether the configuration is valid or not
    uint8_t  control_device; // LEFT_CONTROL_DEVICE or RIGHT_CONTROL_DEVICE if valid
};

//////////////////// FUNCTIONS ////////////////////////
#ifdef __cplusplus
extern "C" {
#endif

 /**
*  Reads the Teensy's UI Configuration from EEPROM, and returns it.
*  If a valid configuration has been loaded, then the "is_valid"
*  field of the returned struct will be true. Otherwise, is_valid will be 
*  false.
*/
config_settings readConfiguration();


/**
*  Displays a user menu for a given amount of time, to give the user the option
*  to set the Teensy UI's configuration.
*  
*  If the user selects to update the configuration, then this function will construct a
*  config_settings struct from user input over serial, and then invoke writeConfiguration() to 
*  write the settings to EEPROM
*
*  @param timeout_s: The timeout, in seconds, for which to display the menu and listen for user input
*/
void showMenu(int timeout_s);


/**
*  Writes a provided configuration to EEPROM for storage
*  
*  @param config: The config_settings to write to EEPROM
*/
void writeConfiguration(const config_settings config);

/**
* Returns a human-readable string of a configuration
*
* @param config: The config_settings to return a human-readable string of
*/
String makeReadable(const config_settings config);

#ifdef __cplusplus
}
#endif

} // UIConfig



#endif // UI_CONFIGURE_H