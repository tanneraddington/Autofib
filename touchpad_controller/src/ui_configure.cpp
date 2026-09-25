/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jason Shrand Vanderbilt University 2020. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/
#include "ui_configure.h"


UIConfig::config_settings UIConfig::readConfiguration()
{
	UIConfig::config_settings settings;
	settings.is_valid = false;
	settings.control_device = UNDEFINED_CONTROL_DEVICE;

	///// Read Control Device /////
    settings.control_device = EEPROM.read(CONTROL_DEVICE_EEPROM_ADDR);

    /////////////////////////////////
	/////       Validation      /////
	/////////////////////////////////
	if(settings.control_device == UIConfig::LEFT_CONTROL_DEVICE or 
	   settings.control_device == UIConfig::RIGHT_CONTROL_DEVICE)
	{
		settings.is_valid = true;
	}
	// NOTE: Add validation logic for any new config parameters here, and update settings.is_valid

	return settings;
}


//=========================================================================//
//                            showMenu                                     //
//=========================================================================//


// Helper functions
void prompt_control_device(UIConfig::config_settings &settings)
{
	// Prompts user for the control device, and populates the 'control_device' field from the input
	bool is_valid_choice = false;
    uint8_t device_choice = UIConfig::UNDEFINED_CONTROL_DEVICE;
	while(!is_valid_choice)
	{
		Serial.println("Please select a control device configuration.");
		Serial.println("1)  LEFT CONTROL DEVICE");
		Serial.println("2)  RIGHT CONTROL DEVICE");

		while(Serial.available() < 1){} // Wait for user to press a key

		// Clear serial buffer
		while(Serial.available() > 0)
		{
			char c = Serial.read();
			if(c == '1')
			{
			    device_choice = UIConfig::LEFT_CONTROL_DEVICE;
			    is_valid_choice = true;
			}
			else if(c == '2')
			{
				device_choice = UIConfig::RIGHT_CONTROL_DEVICE;
				is_valid_choice = true;
			}
		}
	}

	// A valid choice was selected. Update the settings
	settings.control_device = device_choice;
}

void UIConfig::showMenu(int timeout_s)
{

	// Check for a keypress before the timeout
    bool show_menu = false;
    delay(10);
	Serial.println("Press 'z' to enter Prostate UI Configuration.");

    unsigned long start = millis();
    int elapsed_secs = 0;

    while(elapsed_secs < timeout_s)
    {
    	elapsed_secs = (millis() - start)/1000;
    	if (Serial.available() > 0)
    	{
    		char inChar = Serial.read();
    		if(inChar == 'Z' or inChar == 'z')
    		{
    			show_menu = true;
    			break;
    		}
    	}
    }

    if(show_menu)
    {
    	Serial.println("*************************************************");
        Serial.println("* Welcome to the Prostate UI Configuration menu! *");
        Serial.println("**************************************************");
        Serial.println();

        // Show the current configuration
        Serial.println("Current configuration:");
        UIConfig::config_settings cur_config = readConfiguration();
        String config_str = UIConfig::makeReadable(cur_config);
        Serial.println(config_str);

        //// Prompt user for new configuration ////
        UIConfig::config_settings new_config;
        prompt_control_device(new_config);
        // NOTE: Call additional helper functions here to prompt user for new config parameters
        //      (See prompt_control_device for example)

        new_config.is_valid = true;


        // Update config in EEPROM
        UIConfig::writeConfiguration(new_config);
    }
    else
    {
	   Serial.println("Config option timeout reached.");
	}
}

void UIConfig::writeConfiguration(const UIConfig::config_settings config)
{
	EEPROM.write(CONTROL_DEVICE_EEPROM_ADDR, config.control_device);
	// NOTE: Add additional writes here to save new config parameters
}



String UIConfig::makeReadable(const UIConfig::config_settings config)
{
	String s = "";

	// is_valid flag
	s.concat("Is valid: ");
	s.concat(config.is_valid? "True" : "False");
	s.concat("\n");

	// Control Device
	s.concat("Control Device: ");
	switch(config.control_device)
	{
		case UIConfig::UNDEFINED_CONTROL_DEVICE :
		    s.concat("Undefined");
		    break;
		case UIConfig::LEFT_CONTROL_DEVICE :
		    s.concat("Left");
		    break;
		case UIConfig::RIGHT_CONTROL_DEVICE :
		    s.concat("Right");
		    break;
		default:
		    s.concat("Undefined");
	}
	s.concat("\n");

	// NOTE: Add to-readable logic for new config parameters here

	return s;
}