#ifndef COMBINE_CONTROL_HPP
#define COMBINE_CONTROL_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <fstream>

class CombineControl : public rclcpp::Node
{
    public:
        CombineControl(const rclcpp::NodeOptions & options);
        void csv_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
        std::ofstream csv_file;

    private:
        void callbackL(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
        void callbackR(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
        void publish_combined();

        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_l;
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_r;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub;
        
        std_msgs::msg::Float32MultiArray latest_l;
        std_msgs::msg::Float32MultiArray latest_r;

        rclcpp::TimerBase::SharedPtr timer;

        bool comments;

};

#endif