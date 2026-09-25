#ifndef TOUCHPAD_SERIAL_L_HPP
#define TOUCHPAD_SERIAL_L_HPP

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_srvs/srv/empty.hpp"
#include <optional>
#include <fstream>

class TouchpadNodeL: public rclcpp::Node
{
    public:
        TouchpadNodeL();
        //void csv_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
        //std::ofstream csv_file;

    private:
        //void topic_callback(const geometry_msgs::msg::Twist::SharedPtr msg);

        //Subscriber and Publisher creation
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher_;
        //rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr data_publisher;
        //rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr write_to_csv_subscription;

        rclcpp::TimerBase::SharedPtr timer;
        std::optional<geometry_msgs::msg::Twist> latest_msg;
        //std_msgs::msg::Float32MultiArray data_store;

        // //Service to enable movement after homing
        // rclcpp::Service<std_srvs::srv::Empty>::SharedPtr move_ready_service;

        //Flag to control whether UI messages are sent to motors
        // bool ready_to_move;

        //Callback functions
        void topic_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
        void publish_data();
        // void enable_movement(const std_srvs::srv::Empty::Request::SharedPtr request,
        //                     const std_srvs::srv::Empty::Response::SharedPtr response);


};

#endif