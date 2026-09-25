#include "combine_control.hpp"

// Combines left and right controller commands for motors before sending to motor node

CombineControl::CombineControl(const rclcpp::NodeOptions & options) : Node("combine_control",options)
{
    // Subscribes to left controller output
    sub_l = this->create_subscription<std_msgs::msg::Float32MultiArray>("/robot/state/current_state_l",10,std::bind(&CombineControl::callbackL,this,std::placeholders::_1));
    // Subscribes to right controller output
    sub_r = this->create_subscription<std_msgs::msg::Float32MultiArray>("/robot/state/current_state_r",10,std::bind(&CombineControl::callbackR,this,std::placeholders::_1));
    // Publishes combined left and right controller message to motor node
    pub = this->create_publisher<std_msgs::msg::Float32MultiArray>("/robot/state/current_state",10);
    // Controls publishing rate to not overwhelm motors
    timer = this->create_wall_timer(std::chrono::milliseconds(50),std::bind(&CombineControl::publish_combined,this));
}

void CombineControl::callbackL(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    // Pointer to most recent message to preserve order
    latest_l = *msg;  
    //RCLCPP_INFO(this->get_logger(),"Left: %.2f, %.2f, %.2f, %.2f",latest_l[0],latest_l[1],latest_l[2],latest_l[3]);  
}

void CombineControl::callbackR(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    // Pointer to most recent message to preserve order
    latest_r = *msg;
    //RCLCPP_INFO(this->get_logger(),"Right: %.4f, %.4f, %.4f, %.4f",latest_r[0],latest_r[1],latest_r[2],latest_r[3]);
}

void CombineControl::publish_combined()
{
    if (!latest_l.data.empty() && !latest_r.data.empty())
    {   
        // Combine left and right motor arrays into one array to control left and right tubes simultaneously
        // RCLCPP_INFO(this->get_logger(),"Right: %.4f, %.4f, %.4f, %.4f,%.4f, %.4f, %.4f, %.4f",latest_r.data[0],latest_r.data[1],latest_r.data[2],latest_r.data[3],latest_r.data[4],latest_r.data[5],latest_r.data[6],latest_r.data[7]);
        // RCLCPP_INFO(this->get_logger(),"Left: %.4f, %.4f, %.4f, %.4f,%.4f, %.4f, %.4f, %.4f",combined_msg.data[0],combined_msg.data[1],combined_msg.data[2],combined_msg.data[3],combined_msg.data[4],combined_msg.data[5],combined_msg.data[6],combined_msg.data[7]);
        std_msgs::msg::Float32MultiArray combined_msg;
        combined_msg.data = latest_r.data;
        combined_msg.data[4] = latest_l.data[4];
        combined_msg.data[5] = latest_l.data[5];
        combined_msg.data[6] = latest_l.data[6];
        combined_msg.data[7] = latest_l.data[7];
        RCLCPP_INFO(this->get_logger(),"Left: %.4f, %.4f, %.4f, %.4f,%.4f, %.4f, %.4f, %.4f",combined_msg.data[0],combined_msg.data[1],combined_msg.data[2],combined_msg.data[3],combined_msg.data[4],combined_msg.data[5],combined_msg.data[6],combined_msg.data[7]);
        // combined_msg.data[4] = 0.0;
        // combined_msg.data[5] = 0.0;
        // combined_msg.data[6] = 0.0;
        // combined_msg.data[7] = 0.0;

        pub->publish(combined_msg);
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc,argv);
    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true);
    rclcpp::spin(std::make_shared<CombineControl>(options));
    rclcpp::shutdown();
    return 0;
}