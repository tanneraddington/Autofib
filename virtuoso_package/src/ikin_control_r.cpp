#include "ikin_control_r.hpp"
#include <cmath>
#include <iostream>

//Takes [x,y,z] position, converts to joint values using closed-form inverse kinematics, and sends to motors in Float32MultiArray

#define CURVATURE 100

ikinControlR::ikinControlR(const rclcpp::NodeOptions & options):Node("ikin_control_r",options)
{
  //Change subscriber here to take in x,y,z coordinates to move to
  subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/xyz_r",10,std::bind(&ikinControlR::topic_callback,this,std::placeholders::_1));
  //Publishes to node to combine left and right motor arrays
  publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/robot/state/current_state_r",10);
  
  data_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/output_data_right",10);

  /////Include this if you want to write controller input data to a csv/////
  // write_to_csv_subscription = this->create_subscription<std_msgs::msg::Float32MultiArray>("/output_data_left",10,std::bind(&VirtuosoUINodeL::csv_callback,this,std::placeholders::_1));
  // csv_file.open("output_data_left.csv",std::ios::out | std::ios::trunc);
  // if (csv_file.is_open()) {
  //   csv_file << "Timestamp,Data\n";
  // } else {
  //   RCLCPP_ERROR(this->get_logger(),"Failed to open CSV file");
  // }
  
  //Controls publishing rate to not overwhelm motor node
  //timer = this->create_wall_timer(std::chrono::milliseconds(20), std::bind(&ikinControlL::publish_data,this));

  RCLCPP_INFO(this->get_logger(),"Right iKin controller has started!");
  data_store.data = {0.0};
}

/////Include this if you want to write controller input data to a csv/////
// void VirtuosoUINodeL::csv_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
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

void ikinControlR::topic_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
  latest_msg = *msg;    // pointer to most recent message to make sure order of messages is preserved with timer
  publish_data();
}

void ikinControlR::publish_data()
{
    if (!latest_msg) return;

    //RCLCPP_INFO(this->get_logger(),"Received. Yaw = %.2f, Pitch = %.2f, Roll = %.2f, Linear z = %.2f",
    //latest_msg->linear.x, latest_msg->linear.y, latest_msg->linear.z, latest_msg->angular.z);
    
    //Create new message structured for motor node
    auto new_msg = std_msgs::msg::Float32MultiArray();

    //Initialize array layout
    new_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
    new_msg.layout.dim[0].label = "state_values";
    new_msg.layout.dim[0].size = 8;   // 8 motors (gripper & camera controlled in motor node)
    new_msg.data.resize(9);   //not necessary, ensures proper size

    // // Scale UI inputs from -1 to 1 -> angles/distances that represent physical limitations of controller
    // // double inner_rot = latest_msg->angular.z * M_PI;          // can rotate from -8pi to 8pi (bigger than right for corkscrew)
    // double inner_rot = latest_msg->linear.z * M_PI;          // can rotate from -8pi to 8pi (bigger than right for corkscrew)
    // // double inner_trans = ((latest_msg->linear.z) + 1.0)*45.0/2000;  // can translate from 0 to 55mm (arduino output scaled -1 to 1)
    // double inner_trans = (1.0 - (latest_msg->angular.z))*41.0/1000;  // can translate from 0 to 55mm (arduino output scaled -1 to 1)
    // double side = latest_msg->linear.x * M_PI / 3;            // can rotate from -pi/3 to pi/3
    // // double up_down = abs(latest_msg->linear.y) * 3 * M_PI / 4;         // can rotate from -pi/2 to pi/2
    // double up_down = latest_msg->linear.y * M_PI / 2;         // can rotate from -pi/2 to pi/2
    // if (up_down < 0.0) {
    //   up_down = 0.0;
    // }
    double x = latest_msg->data[0];
    double y = latest_msg->data[1];
    double z = latest_msg->data[2];

    double r = sqrt(pow(x,2) + pow(y,2));
    double alpha = atan2(y,x);
    // double d_outer = 1/CURVATURE * acos(1-CURVATURE*r);
    // double d_inner = d_outer + ((z-sin(CURVATURE*d_outer))/CURVATURE)/cos(CURVATURE*d_outer);
    double inner_rot = 0.0;
    double u_tilde = r - (1/CURVATURE);
    double w_tilde = z;
    double diff = pow(r,2) + pow(z,2) - (2*r/CURVATURE);
    if (diff < 0.0) {
      diff = 0.0;
    }
    double delta = sqrt(diff);
    double gamma = atan2(w_tilde,u_tilde);
    double beta = atan2(delta,(-1/CURVATURE));
    double chi = beta - gamma;
    double d_outer = chi / CURVATURE;
    double d_inner = d_outer + delta;

    //Right controller controlled by separate node
    new_msg.data[4] = 0.0;                                       
    new_msg.data[5] = 0.0;   
    new_msg.data[7] = 0.0;            
    new_msg.data[6] = 0.0;     
    //Left controller -> joint values              
    new_msg.data[0] = inner_rot;        //inner rotation
    new_msg.data[1] = alpha;            //outer rotation
    new_msg.data[3] = d_outer;          //outer translation 
    new_msg.data[2] = d_inner;          //inner translation (cannot go inside outer tube)

    //Display motion to user (positive rotation = CCW and positive translation = out from robot)
    //if (comments) {
      RCLCPP_INFO(this->get_logger(),"Inner rot = %.2f deg, Outer rot = %.2f deg, Inner trans = %.2f mm, Outer trans = %.2f mm",
      new_msg.data[0], new_msg.data[1], new_msg.data[2]*-1000, new_msg.data[3]*-1000);

    //}

    publisher_->publish(new_msg);

    // //Store controller up/down input for tube tracking
    // data_store.data[0] = up_down;
    // //Publish to topic that tube tracking listens to (in motor node)
    // data_publisher->publish(data_store);

}

int main(int argc, char *argv[])
{
  rclcpp::init(argc,argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true);
  
  auto node = std::make_shared<ikinControlR>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  /////Include if want to save controller data to csv/////
  // if (node->csv_file.is_open()) {
  //   node->csv_file.close();
  // }

  return 0;
}