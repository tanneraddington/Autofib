/*  Force Sensor Reading code with ROS
************************************************
************************************************
************************************************
************************************************
*/

// to run arduino with ros using:
//     rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0 _baud:=115200


#include <ros.h> //enable ROS communications
#include <geometry_msgs/Vector3.h> //ROS 3d vector msg type
#include <std_msgs/Bool.h> //ROS bool msg type
#include "HX711.h"

HX711 scale1;

// Set up ROS node and publisher for coil temps and thermal shutdown activated
geometry_msgs::Vector3 force_msg;
ros::Publisher pub_field("force_read", &force_msg);

ros::NodeHandle nh;


// Pins for over-temperature, generic fault, and master shutdown pin
#define data_pin1  A5
#define clock_pin1  A4

#define data_pin2  5
#define clock_pin2  4

#define data_pin3  7
#define clock_pin3  6

#define printSerial true

float calibration_factor = -531000.0; // calibrated to kg

// Temperature Readings Variables and Sensor Faults. Faults initialized to no errors.
double force1 = 0.0;


void setup() {
  Serial.begin(115200);
  delay(1000); //This delay ensures successful startup when using low-quality power supplies
  //  nh.getHardware()->setBaud(9600);

  //  long zero_factor = scale.read_average(); //Get a baseline reading
  //

  // Begin each coil thermocouple sensor board communication
  scale1.begin(data_pin1, clock_pin1);

  scale1.set_scale(calibration_factor);
  scale1.tare(); //Reset the scale to 0

  //Initialize ROS Node and add two publishers
  nh.initNode();
  nh.advertise(pub_field);
  Serial.println("TEST");
}



void loop() {

  // Record new readings from thermocouples
  force1  = scale1.get_units();

  // publish force message
  if (nh.connected()) {
    force_msg.x = force1;
    pub_field.publish(&force_msg);
  }

  if (printSerial) {
    //    Serial.print(millis());
    //    Serial.print(", ");
    Serial.println(force1);
  }


  nh.spinOnce();
  delay(10);
}
