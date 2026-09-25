#include "touchpad_serial_r.hpp"
#include <cmath>
#include <iostream>

//Takes twist input from Touchpad UI, converts to joint values using geometry, and sends to motors in Float32MultiArray

#define CURVATURE 70

TouchpadNodeR::TouchpadNodeR():Node("touchpad_node_r")
{
  //Subscriber to receive twist from UI
  subscription_ = this->create_subscription<geometry_msgs::msg::Twist>("/ui_twist_r",10,std::bind(&TouchpadNodeR::topic_callback,this,std::placeholders::_1));
  //Publishes to node to combine left and right motor arrays
  publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/robot/state/current_state",10);
  
  //data_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/output_data_right",10);

//   write_to_csv_subscription = this->create_subscription<std_msgs::msg::Float32MultiArray>("/output_data_right",10,std::bind(&VirtuosoUINodeR::csv_callback,this,std::placeholders::_1));
//   csv_file.open("output_data_right.csv",std::ios::out | std::ios::trunc);
//   if (csv_file.is_open()) {
//     csv_file << "Timestamp,Data\n";
//   } else {
//     RCLCPP_ERROR(this->get_logger(),"Failed to open CSV file");
//   }
  
  //Controls publishing rate to not overwhelm motor node
  timer = this->create_wall_timer(std::chrono::milliseconds(20), std::bind(&TouchpadNodeR::publish_data,this));

  RCLCPP_INFO(this->get_logger(),"Right Touchpad Controller Interpreter has been started!");
  //data_store.data = {0.0};
}

// void VirtuosoUINodeR::csv_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
//   auto now = std::chrono::system_clock::now();
//   auto duration = now.time_since_epoch();
//   double time_in_sec = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
//   std::string time_str = std::to_string(time_in_sec);
//   std::string data_str = "[";
//   for (size_t i = 0; i < msg->data.size(); ++i) {
//     data_str += std::to_string(msg->data[i]);
//     if (i < msg->data.size()-1) {
//       data_str += ", ";
//     }
//   }
//   data_str += "]";

//   if (csv_file.is_open()) {
//     csv_file << time_str << "," << data_str << "\n";
//   }
// }


void TouchpadNodeR::topic_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
  latest_msg = *msg;    // pointer to most recent message to make sure order of messages is preserved with timer
}

void TouchpadNodeR::publish_data()
{

    if (!latest_msg) return;

    RCLCPP_INFO(this->get_logger(),"Received. Yaw = %.2f, Pitch = %.2f, Roll = %.2f, Linear z = %.2f",
    latest_msg->linear.x, latest_msg->linear.y, latest_msg->linear.z, latest_msg->angular.z);
    
    //Create new message structured for motor node
    auto new_msg = std_msgs::msg::Float32MultiArray();

    //Initialize array layout
    new_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
    new_msg.layout.dim[0].label = "state_values";
    new_msg.layout.dim[0].size = 9;
    //new_msg.layout.dim[0].stride = 8;
    new_msg.data.resize(8);   //not necessary, ensures proper size

    // Scale UI inputs from -1 to 1 -> angles that represent physical limitations of controller
    //double inner_rot = latest_msg->angular.z * M_PI;             // can rotate from -pi to pi
    double inner_rot = 0;                                           // inner tube rotation not controlled
    double inner_trans = -((latest_msg->linear.z))*20/1000;            // can translate from 0 to 20mm
    double side = latest_msg->linear.x * M_PI / 3;                // can rotate from -pi/3 to pi/3
    double up_down = ((latest_msg->linear.y)) * M_PI / 2;             // can rotate from -pi/2 to pi/2

    //Right controller -> joint values
    new_msg.data[0] = -inner_rot;                                                                   //inner rotation 1:1 mapping
    //new_msg.data[1] = (atan2(sin(side)*cos(up_down),sin(up_down)*cos(side)));                         //outer rotation 
    new_msg.data[1] = (atan2(cos(side)*sin(up_down),cos(up_down)*sin(side)));                         //outer rotation 
    new_msg.data[3] = -sqrt(pow(sin(side)*cos(up_down),2) + pow(sin(up_down)*cos(side),2)) * .042;   //outer translation
    new_msg.data[2] = (inner_trans + new_msg.data[3]);                                               //inner translation (cannot go inside outer tube)
    //Left controller controlled by separate node
    new_msg.data[4] = 0.0;
    new_msg.data[5] = 0.0;
    new_msg.data[6] = 0.0;
    new_msg.data[7] = 0.0;

    //Display motion to user (positive rotation = ** and positive translation = out from robot)
    RCLCPP_INFO(this->get_logger(),"Inner rot = %.2f deg, Outer rot = %.2f deg, Inner trans = %.2f mm, Outer trans = %.2f mm",
    new_msg.data[0], new_msg.data[1]*180/M_PI, new_msg.data[2]*-1000, new_msg.data[3]*-1000);


    publisher_->publish(new_msg);

    //data_store.data[0] = up_down;
    //data_publisher->publish(data_store);

}

int main(int argc, char *argv[])
{
  rclcpp::init(argc,argv);
  
  auto node = std::make_shared<TouchpadNodeR>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  
//   if (node->csv_file.is_open()) {
//     node->csv_file.close();
//   }

  return 0;
}