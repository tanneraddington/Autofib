#ifndef VIRTUOSO_UI_NODE_L_HPP
#define VIRTUOSO_UI_NODE_L_HPP

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_srvs/srv/empty.hpp"
#include <optional>
#include <fstream>

class VirtuosoUINodeL: public rclcpp::Node
{
    public:
        VirtuosoUINodeL(const rclcpp::NodeOptions & options);
        void csv_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
        std::ofstream csv_file;

    private:
        //void topic_callback(const geometry_msgs::msg::Twist::SharedPtr msg);

        //Subscriber and Publisher creation
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr data_publisher;
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr write_to_csv_subscription;

        rclcpp::TimerBase::SharedPtr timer;
        std::optional<geometry_msgs::msg::Twist> latest_msg;
        std_msgs::msg::Float32MultiArray data_store;

        //Callback functions
        void topic_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
        void publish_data();

        bool comments;
        double last_side = 0.0;


};

#endif