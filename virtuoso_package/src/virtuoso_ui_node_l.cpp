#include "virtuoso_ui_node_l.hpp"
#include <cmath>
#include <iostream>

//Takes twist input from Virtuoso UI, converts to joint values using geometry, and sends to motors in Float32MultiArray

#define CURVATURE 89.4930
#define E 8300
#define KO 100
#define KI 0
#define OD_O 0.00145
#define OD_I 0.0008
#define ID_O 0.0011
#define ID_I 0.0005

VirtuosoUINodeL::VirtuosoUINodeL(const rclcpp::NodeOptions & options):Node("virtuoso_ui_node_l",options)
{
  //Subscriber to receive twist from UI
  subscription_ = this->create_subscription<geometry_msgs::msg::Twist>("/ui_twist_l",10,std::bind(&VirtuosoUINodeL::topic_callback,this,std::placeholders::_1));
  //Publishes to node to combine left and right motor arrays
  publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/robot/state/current_state_l",10);
  
  data_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/output_data_left",10);

  /////Include this if you want to write controller input data to a csv/////
  // write_to_csv_subscription = this->create_subscription<std_msgs::msg::Float32MultiArray>("/output_data_left",10,std::bind(&VirtuosoUINodeL::csv_callback,this,std::placeholders::_1));
  // csv_file.open("output_data_left.csv",std::ios::out | std::ios::trunc);
  // if (csv_file.is_open()) {
  //   csv_file << "Timestamp,Data\n";
  // } else {
  //   RCLCPP_ERROR(this->get_logger(),"Failed to open CSV file");
  // }
  
  //Controls publishing rate to not overwhelm motor node
  timer = this->create_wall_timer(std::chrono::milliseconds(20), std::bind(&VirtuosoUINodeL::publish_data,this));

  RCLCPP_INFO(this->get_logger(),"Left Virtuoso Controller Interpreter has started!");
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

void VirtuosoUINodeL::topic_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
  latest_msg = *msg;    // pointer to most recent message to make sure order of messages is preserved with timer
}

void VirtuosoUINodeL::publish_data()
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
    new_msg.data.resize(8);   //not necessary, ensures proper size

    //////////////////////////WORKING VERSION ANGLE MAPPING///////////////////////////////////
    // Scale UI inputs from -1 to 1 -> angles/distances that represent physical limitations of controller
    // double inner_rot = latest_msg->angular.z * M_PI;          // can rotate from -8pi to 8pi (bigger than right for corkscrew)
    double inner_rot = latest_msg->linear.z * M_PI;          // can rotate from -8pi to 8pi (bigger than right for corkscrew)
    // double inner_trans = ((latest_msg->linear.z) + 1.0)*45.0/2000;  // can translate from 0 to 45mm (arduino output scaled -1 to 1)
    double inner_trans = ((latest_msg->angular.z) + 1.0 )*45.0/2000;  // can translate from 0 to 45mm (arduino output scaled -1 to 1)
    double side = latest_msg->linear.x * M_PI / 3;            // can rotate from -pi/2 to pi/2
    // double side;
    // if (side_raw - last_side > 0.05) {
    //   side = side_raw + M_PI/18;
    // }
    // else if (side_raw - last_side < 0.05) {
    //   side = side_raw - M_PI/18;
    // }
    // else {
    //   side = side_raw;
    // }
    // last_side = side;
    // double up_down = abs(latest_msg->linear.y) * 3 * M_PI / 4;         // can rotate from -pi/2 to pi/2
    double up_down = latest_msg->linear.y * M_PI / 2;         // can rotate from -pi/2 to pi/2
    if (up_down < 0.0) {
      up_down = 0.0;
    }

    //////////////////////////////////////////////////////////////////////////////////////

    // double Li = 0.045;
    // double Lo = 0.04;
    // double z_max = Li;

    // // Scale UI inputs from -1 to 1 -> max values in cartesian space
    // double inner_rot = latest_msg->linear.z * M_PI;          // can rotate from -2pi to 2pi (bigger than right for corkscrew)
    // // double inner_trans = ((latest_msg->linear.z) + 1.0)*45.0/2000;  // can translate from 0 to 45mm (arduino output scaled -1 to 1)
    // double inner_trans = ((latest_msg->angular.z) + 1.0 )*z_max/2;  // can translate from 0 to 45mm (arduino output scaled -1 to 1)

    // // double isin1 = asin(CURVATURE*inner_trans/sqrt(1 + pow(Li*CURVATURE,2)));
    // double arg = CURVATURE * inner_trans / sqrt(1 + pow(Li*CURVATURE,2));
    // arg = std::clamp(arg, -1.0, 1.0);
    // double isin1 = asin(arg);
    // double isin2 = M_PI - isin1;
    // double do1 = 1/CURVATURE * (isin1 - atan(Li*CURVATURE));
    // double do2 = 1/CURVATURE * (isin2 - atan(Li*CURVATURE));
    // bool valid1 = (do1 >= 0.0001) && (do1 <= Lo);
    // bool valid2 = (do2 >= 0.0001) && (do2 <= Lo);
    // double d_o = 0.0;
    // if (valid1 && !valid2) {
    //   d_o = do1;
    // }
    // else if (valid2 && !valid1) {
    //   d_o = do2;
    // }
    // else {
    //   RCLCPP_INFO(this->get_logger(),"Invalid solution");
    // }

    // //double rmax = (1/CURVATURE * (1-cos(CURVATURE*d_o))) + Li*sin(CURVATURE*d_o);
    // double rmax = 0.0223;
    // double x_max = rmax;
    // double y_max = rmax;

    // double side_raw = latest_msg->linear.x * x_max;            // can rotate from -pi/2 to pi/2
    // // double up_down = abs(latest_msg->linear.y) * 3 * M_PI / 4;         // can rotate from -pi/2 to pi/2
    // double up_down_raw = latest_msg->linear.y * y_max;         // can rotate from -pi/2 to pi/2
    // if (up_down_raw < 0.0) {
    //   up_down_raw = 0.0;
    // }
    // double side = up_down_raw;
    // double up_down = -side_raw;

    // double cam_angle = 45.0 * M_PI/180.0;
    // double x = cos(cam_angle) * side + sin(cam_angle) * inner_trans;
    // double y = up_down;
    // double z = -sin(cam_angle) * side + cos(cam_angle) * inner_trans;

    // // RCLCPP_INFO(this->get_logger(),"x: %.3f, y: %.3f, z:  %.3f",side, up_down,inner_trans);

    /////////////////////////////////////////////////////////////////////////////////

    //Right controller controlled by separate node
    new_msg.data[0] = 0.0;                                       
    new_msg.data[1] = 0.0;   
    new_msg.data[3] = 0.0;            
    new_msg.data[2] = 0.0;  

    ///////////////////////////WORKING VERSION ANGLE MAPPING///////////////////////////////////////
       
    if (up_down < M_PI/3) {
    //Left controller -> joint values              
      new_msg.data[4] = inner_rot;                                                                           //inner rotation 1:1 mapping
      new_msg.data[5] = -atan2(sin(side)*cos(up_down),sin(up_down)*cos(side));                               //outer rotation
      new_msg.data[7] = sqrt(pow(sin(side)*cos(up_down),2) + pow(sin(up_down)*cos(side),2)) * .04;           //outer translation (max of 45mm)
      new_msg.data[6] = inner_trans + (new_msg.data[7]);                                                     //inner translation (cannot go inside outer tube)
    }

    // 1:1 outer rot mapping for high outer tube translation values (avoids 0 values of atan2 limiting rotation)
    else {
      //Left controller -> joint values              
      new_msg.data[4] = inner_rot;                                                                           //inner rotation 1:1 mapping
      new_msg.data[5] = -side;                                                                                //outer rotation 1:1 mappint
      // new_msg.data[7] = sqrt(pow(sin(side)*cos(up_down),2) + pow(sin(up_down)*cos(side),2)) * .04;          //outer translation (max of 45mm)
      new_msg.data[7] = up_down/(M_PI/2) * .04;
      new_msg.data[6] = inner_trans + (new_msg.data[7]);                                                    // inner translation (cannot go inside outer tube)
    }

    //Display motion to user (positive rotation = CCW and positive translation = out from robot)
    if (comments) {
      RCLCPP_INFO(this->get_logger(),"Inner rot = %.2f deg, Outer rot = %.2f deg, Inner trans = %.2f mm, Outer trans = %.2f mm",
      new_msg.data[4], new_msg.data[5], new_msg.data[6]*-1000, new_msg.data[7]*-1000);

    }

    publisher_->publish(new_msg);

    ///////////////////////////////////////////////////////////////////////////////////////////

    // double I_o = M_PI/64 * (pow(OD_O,4) - pow(ID_O,4));
    // double I_i = M_PI/64 * (pow(OD_I,4) - pow(ID_I,4));
    // double r = sqrt(pow(x,2) + pow(y,2));
    // double B = sqrt(pow(z*CURVATURE,2) + pow((r*CURVATURE)-1,2));
    // double phi = atan2((r*CURVATURE-1),z*CURVATURE);

    // //Left controller -> joint values              
    // new_msg.data[4] = inner_rot;                                                                           //inner rotation 1:1 mapping
    // new_msg.data[5] = atan2(y,x);                                                                 //outer rotation 
    // // double outer_trans = (phi + asin(1/B))/CURVATURE;   
    // double invB = 1.0 / B;
    // invB = std::clamp(invB, -1.0, 1.0);
    // double outer_trans = (phi + asin(invB)) / CURVATURE;                                                      // outer translation
    // new_msg.data[7] = outer_trans;
    // double denom = CURVATURE * sin(outer_trans * CURVATURE);
    // if (fabs(denom) < 1e-6) denom = 1e-6 * (denom >= 0 ? 1 : -1);
    // // double delta = (cos(CURVATURE*outer_trans) + r*CURVATURE - 1)/(CURVATURE*sin(outer_trans*CURVATURE));                                                                 
    // double delta = (cos(CURVATURE*outer_trans) + r*CURVATURE - 1)/denom;
    // new_msg.data[6] = outer_trans + delta;                                                    // inner translation (cannot go inside outer tube)

    /////////////////////////////////////////////////////////////////////////////////////////////

    //Display motion to user (positive rotation = CCW and positive translation = out from robot)
    if (comments) {
      RCLCPP_INFO(this->get_logger(),"Inner rot = %.2f deg, Outer rot = %.2f deg, Inner trans = %.2f mm, Outer trans = %.2f mm",
      new_msg.data[4], new_msg.data[5], new_msg.data[6]*-1000, new_msg.data[7]*-1000);

    }
    // if ((delta > 0.0) && (outer_trans + delta < 0.09)) 
    // {
    publisher_->publish(new_msg);
    //}
    //Store controller up/down input for tube tracking
    data_store.data[0] = up_down;
    //Publish to topic that tube tracking listens to (in motor node)
    data_publisher->publish(data_store);

}

int main(int argc, char *argv[])
{
  rclcpp::init(argc,argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true);
  
  auto node = std::make_shared<VirtuosoUINodeL>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  /////Include if want to save controller data to csv/////
  // if (node->csv_file.is_open()) {
  //   node->csv_file.close();
  // }

  return 0;
}