/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jason Shrand, Dominick Ropella, Vanderbilt University 2020. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/
#include "touchpad.h"


////////////////////////// PUBLIC METHODS //////////////////////////
Touchpad::Touchpad(uint8_t CS_pin, uint8_t DR_pin, uint8_t LED_pin, bool is_curved_overlay)
{
	padData.CS_Pin = CS_pin;
	padData.DR_Pin = DR_pin;
	padData.LED_Pin = LED_pin;
	isCurvedOverlay = is_curved_overlay;
}

void Touchpad::initialize()
{
    Pinnacle_Init();

    setAdcAttenuation(ADC_ATTENUATE_2X);
    tuneEdgeSensitivity();
    Pinnacle_forceCalibration();

    Pinnacle_EnableFeed(true);
}

void Touchpad::readTouchData(touchinfo_t &touchInfo)
{
	// if(DR_Asserted())
	// {
	// 	// There is new data from the touchpad

	//     // Get the latest data from the pinnacle, check if it's valid, and scale it.


  //       if (currentData.hovering == 0) 
  //       {  
  //       	// Touch is not hovering

  //           if (!touch_trigger)
  //           {
  //           	// This is a new touch
  //               x_prev = currentData.xValue;
  //               y_prev = currentData.yValue;
  //           }

  //           // Add the difference from the previous touch
  //           diff_x = double(x_prev - currentData.xValue);
  //           diff_y = double(y_prev - currentData.yValue);          
  //           x_disp = x_disp + 0.001 * diff_x;
  //           y_disp = y_disp +0.001 * diff_y;
      
  //           x_prev = currentData.xValue;
  //           y_prev = currentData.yValue;
  //           touch_trigger = true;
  //       }
	// }
	// else
	// {
	// 	// No data from touchpad
	// 	touch_trigger = false;
	// }


  if(DR_Asserted())
	{ 	
    Pinnacle_GetAbsolute(&currentData);
	  Pinnacle_CheckValidTouch(&currentData);
	  ScaleData(&currentData, 1024, 1024);

    // Populate the touchinfo struct
    touchInfo.is_valid    = !currentData.hovering;
    touchInfo.x_pos       = currentData.xValue;
    touchInfo.y_pos       = currentData.yValue;
    touchInfo.x_disp      = x_disp; 
    touchInfo.y_disp      = y_disp;
    touchInfo.is_hovering = currentData.hovering; 

    x_prev = currentData.xValue;
    y_prev = currentData.yValue;


  }
  else{
    touchInfo.is_valid    = !currentData.hovering;
    touchInfo.x_pos       = x_prev;
    touchInfo.y_pos       = y_prev;
  }


  String txt;
  touchDataToString(touchInfo, txt);
  Serial.println(txt);
 


	// Update the touchpad LED when the touch is down
	AssertSensorLED(currentData.touchDown);
}

void Touchpad::touchDataToString(const touchinfo_t touchInfo, String &str)
{
	str = ""; // Start with a blank string
	str.concat("Is valid: ");
	if(touchInfo.is_valid)
	{
		str.concat("True");
	}
	else
	{
		str.concat("False");
	}

	str.concat("\nx pos: ");
	str.concat(touchInfo.x_pos);
	str.concat("\ny pos: ");
	str.concat(touchInfo.y_pos);
	str.concat("\nx disp: ");
	str.concat(touchInfo.x_disp);
	str.concat("\ny disp: ");
	str.concat(touchInfo.y_disp);
	
	str.concat("\nIs hovering: ");
	if(touchInfo.is_hovering)
	{
		str.concat("True\n");
	}
	else
	{
		str.concat("False\n");
	}
}



///////////////////////// PRIVATE METHODS //////////////////////////

///// Pinnacle Functions /////
void Touchpad::Pinnacle_Init()
{
  RAP_Init();
  DeAssert_CS();
  pinMode(padData.DR_Pin, INPUT);

  // Host clears SW_CC flag
  Pinnacle_ClearFlags();

  // Host configures bits of registers 0x03 and 0x05
  RAP_Write(0x03, Touchpad::SYSCONFIG_1);
  RAP_Write(0x05, Touchpad::FEEDCONFIG_2);

  // Host enables preferred output mode (absolute)
  RAP_Write(0x04, Touchpad::FEEDCONFIG_1);

  // Host sets z-idle packet count to 5 (default is 30)
  RAP_Write(0x0A, Touchpad::Z_IDLE_COUNT);
  Serial.println("Pinnacle Initialized...");
}

// Reads XYZ data from Pinnacle registers 0x14 through 0x17
// Stores result in absData_t struct with xValue, yValue, and zValue members
void Touchpad::Pinnacle_GetAbsolute(Touchpad::absData_t * result)
{
  uint8_t data[6] = { 0,0,0,0,0,0 };
  RAP_ReadBytes(0x12, data, 6);
  Pinnacle_ClearFlags();

  result->buttonFlags = data[0] & 0x3F;
  result->xValue = data[2] | ((data[4] & 0x0F) << 8);
  result->yValue = data[3] | ((data[4] & 0xF0) << 4);
  result->zValue = data[5] & 0x3F;

  result->touchDown = result->xValue != 0;
}

// Forces Pinnacle to re-calibrate, sometimes useful when miss-compensation
// causes touchpad to miss real touches or false touches are tringgering touch events
void Touchpad::Pinnacle_forceCalibration()
{
  uint8_t CalConfig1Value = 0x00;

  Pinnacle_EnableFeed(false);
  RAP_ReadBytes(0x07, &CalConfig1Value, 1);
  CalConfig1Value |= 0x01;
  RAP_Write(0x07, CalConfig1Value);

  do
  {
    RAP_ReadBytes(0x07, &CalConfig1Value, 1);
  }
  while(CalConfig1Value & 0x01);

  Pinnacle_ClearFlags();
}

// Checks touch data to see if it is a z-idle packet (all zeros)
bool Touchpad::Pinnacle_zIdlePacket(absData_t * data)
{
  return data->xValue == 0 && data->yValue == 0 && data->zValue == 0;
}

// Clears Status1 register flags (SW_CC and SW_DR)
void Touchpad::Pinnacle_ClearFlags()
{
  RAP_Write(0x02, 0x00);
}

// Enables/Disables the feed
void Touchpad::Pinnacle_EnableFeed(bool feedEnable)
{
  uint8_t temp;

  RAP_ReadBytes(0x04, &temp, 1);  // Store contents of FeedConfig1 register

  if(feedEnable)
  {
    temp |= 0x01;  // Set Feed Enable bit
    RAP_Write(0x04, temp);
  }
  else
  {
    temp &= ~0x01; // Clear Feed Enable bit
    RAP_Write(0x04, temp);
  }
}

// This function identifies when a finger is "hovering" so your system can choose to ignore it.
// The sensor detects the finger in the space above the sensor. If the finger is on the surface of the sensor the Z value is highest.
// If the finger is a few millimeters above the surface the z value is much lower.
// Adding a curved overlay will allow the finger to be closer in the middle (so a higher z value) but farther
// on the perimeter (so a lower z value).
// With a curved overlay you tune the gain of the system to see a finger on the perimeter of the sensor
// (the finger is farther away).  Unfortunately a finger near the center will be detected above the surface.
// This code will tell you when to ignore that "hovering" finger.
// ZVALUE_MAP[][] stores a lookup table in which you can define the Z-value and XY position that is considered "hovering". Experimentation/tuning is required.
// NOTE: Z-value output decreases to 0 as you move your finger away from the sensor, and it's maximum value is 0x63 (6-bits).
void Touchpad::Pinnacle_CheckValidTouch(absData_t * touchData)
{
  uint32_t zone_x, zone_y;
  //eliminate hovering
  zone_x = touchData->xValue / ZONESCALE;
  zone_y = touchData->yValue / ZONESCALE;
  touchData->hovering = !(touchData->zValue > ZVALUE_MAP[zone_y][zone_x]);
}

/*  RAP Functions */
void Touchpad::RAP_Init()
{
  pinMode(padData.CS_Pin, OUTPUT);
  SPI.begin();
}

// Reads <count> Pinnacle registers starting at <address>
void Touchpad::RAP_ReadBytes(byte address, byte * data, byte count)
{
  byte cmdByte = READ_MASK | address;   // Form the READ command byte

  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  Assert_CS();
  SPI.transfer(cmdByte);  // Signal a RAP-read operation starting at <address>
  SPI.transfer(0xFC);     // Filler byte
  SPI.transfer(0xFC);     // Filler byte
  for(byte i = 0; i < count; i++)
  {
    data[i] =  SPI.transfer(0xFC);  // Each subsequent SPI transfer gets another register's contents
  }
  DeAssert_CS();

  SPI.endTransaction();
}

// Writes single-byte <data> to <address>
void Touchpad::RAP_Write(byte address, byte data)
{
  byte cmdByte = WRITE_MASK | address;  // Form the WRITE command byte

  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  Assert_CS();
  SPI.transfer(cmdByte);  // Signal a write to register at <address>
  SPI.transfer(data);    // Send <value> to be written to register
  DeAssert_CS();

  SPI.endTransaction();
  delayMicroseconds(DEFAULT_WRITE_DELAY);
}

/*  I/O Functions */
void Touchpad::Assert_CS()
{
  digitalWrite(padData.CS_Pin, LOW);
}

void Touchpad::DeAssert_CS()
{
  digitalWrite(padData.CS_Pin, HIGH);
}

void Touchpad::AssertSensorLED(bool state)
{
  digitalWrite(padData.LED_Pin, !state);
}

bool Touchpad::DR_Asserted()
{
  return digitalRead(padData.DR_Pin);
}

/*  ERA (Extended Register Access) Functions  */
// Reads <count> bytes from an extended register at <address> (16-bit address),
// stores values in <*data>
void Touchpad::ERA_ReadBytes(uint16_t address, uint8_t * data, uint16_t count)
{
  uint8_t ERAControlValue = 0xFF;
  Pinnacle_EnableFeed(false); // Disable feed

  RAP_Write(0x1C, (uint8_t)(address >> 8));     // Send upper byte of ERA address
  RAP_Write(0x1D, (uint8_t)(address & 0x00FF)); // Send lower byte of ERA address

  for(uint16_t i = 0; i < count; i++)
  {
    RAP_Write(0x1E, 0x05);  // Signal ERA-read (auto-increment) to Pinnacle

    // Wait for status register 0x1E to clear
    do
    {
      RAP_ReadBytes(0x1E, &ERAControlValue, 1);
    } while(ERAControlValue != 0x00);

    // Read register to verify that the new value is there.
    RAP_ReadBytes(0x1B, data + i, 1);
    Pinnacle_ClearFlags();
  }
}

// Writes a byte, <data>, to an extended register at <address> (16-bit address)
void Touchpad::ERA_WriteByte(uint16_t address, uint8_t data)
{
  uint8_t ERAControlValue = 0xFF;
  Pinnacle_EnableFeed(false); // Disable feed

  RAP_Write(0x1B, data);      // Send data byte to be written

  RAP_Write(0x1C, (uint8_t)(address >> 8));     // Upper byte of ERA address
  RAP_Write(0x1D, (uint8_t)(address & 0x00FF)); // Lower byte of ERA address

  RAP_Write(0x1E, 0x02);  // Signal an ERA-write to Pinnacle

  // Wait for status register 0x1E to clear
  do
  {
    RAP_ReadBytes(0x1E, &ERAControlValue, 1);
  } while(ERAControlValue != 0x00);

  delayMicroseconds(DEFAULT_WRITE_DELAY);
  Pinnacle_ClearFlags();
}

/*  Curved Overlay Functions  */

// Adjusts the feedback in the ADC, effectively attenuating the finger signal
// By default, the the signal is maximally attenuated (ADC_ATTENUATE_4X for use with thin, flat overlays
void Touchpad::setAdcAttenuation(uint8_t adcGain)
{
  uint8_t temp = 0x00;

  Serial.println();
  Serial.println("Setting ADC Gain...");

  ERA_ReadBytes(0x0187, &temp, 1);

  Serial.print("Current value:\t");
  Serial.println(temp, HEX);

  temp &= 0x3F;         // clear top two bits
  temp |= adcGain;      // set top two bits to the desired ADC Gain Value
  ERA_WriteByte(0x0187, temp);
  ERA_ReadBytes(0x0187, &temp, 1);

  Serial.print("New value:\t");
  Serial.print(temp, HEX);

  switch(adcGain)
  {
    case ADC_ATTENUATE_1X:
      Serial.println(" (X/1)");
      break;
    case ADC_ATTENUATE_2X:
      Serial.println(" (X/2)");
      break;
    case ADC_ATTENUATE_3X:
      Serial.println(" (X/3)");
      break;
    case ADC_ATTENUATE_4X:
      Serial.println(" (X/4)");
      break;
    default:
      break;
  }
}

// Changes thresholds to improve detection of fingers
void Touchpad::tuneEdgeSensitivity()
{
  uint8_t temp = 0x00;

  Serial.println();
  Serial.println("Setting xAxis.WideZMin...");
  ERA_ReadBytes(0x0149, &temp, 1);
  Serial.print("Current value:\t");
  Serial.println(temp, HEX);
  ERA_WriteByte(0x0149,  0x04);
  ERA_ReadBytes(0x0149, &temp, 1);
  Serial.print("New value:\t");
  Serial.println(temp, HEX);

  Serial.println();
  Serial.println("Setting yAxis.WideZMin...");
  ERA_ReadBytes(0x0168, &temp, 1);
  Serial.print("Current value:\t");
  Serial.println(temp, HEX);
  ERA_WriteByte(0x0168,  0x03);
  ERA_ReadBytes(0x0168, &temp, 1);
  Serial.print("New value:\t");
  Serial.println(temp, HEX);
}

///// Logical Scaling Functions /////

// Clips raw coordinates to "reachable" window of sensor
// NOTE: values outside this window can only appear as a result of noise
void Touchpad::ClipCoordinates(absData_t * coordinates)
{
  if(coordinates->xValue < PINNACLE_X_LOWER)
  {
    coordinates->xValue = PINNACLE_X_LOWER;
  }
  else if(coordinates->xValue > PINNACLE_X_UPPER)
  {
   coordinates->xValue = PINNACLE_X_UPPER;
  }
  if(coordinates->yValue < PINNACLE_Y_LOWER)
  {
    coordinates->yValue = PINNACLE_Y_LOWER;
  }
  else if(coordinates->yValue > PINNACLE_Y_UPPER)
  {
    coordinates->yValue = PINNACLE_Y_UPPER;
  }
}

// Scales data to desired X & Y resolution
void Touchpad::ScaleData(absData_t * coordinates, uint16_t xResolution, uint16_t yResolution)
{
  uint32_t xTemp = 0;
  uint32_t yTemp = 0;

  ClipCoordinates(coordinates);

  xTemp = coordinates->xValue;
  yTemp = coordinates->yValue;

  // translate coordinates to (0, 0) reference by subtracting edge-offset
  xTemp -= PINNACLE_X_LOWER;
  yTemp -= PINNACLE_Y_LOWER;

  // scale coordinates to (xResolution, yResolution) range
  coordinates->xValue = (uint16_t)(xTemp * xResolution / PINNACLE_X_RANGE);
  coordinates->yValue = (uint16_t)(yTemp * yResolution / PINNACLE_Y_RANGE);
}