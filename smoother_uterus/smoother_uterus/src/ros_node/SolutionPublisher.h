#pragma once

#include <gtsam/geometry/Pose3.h>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "SmootherPipeline.h"

class SolutionPublisher {
public:
    explicit SolutionPublisher(rclcpp::Node& node);

    void publish_single_arm(const SolvedPipelineSample& solved, const ArmSide side) const;
    void publish_overlay_image(const SolvedPipelineSample& solved) const;
    void publish(const SolvedPipelineSample& solved) const;

private:
    rclcpp::Node& node_;
    std::array<std::array<rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr, 2>, 2> pose_array_pubs_;
    std::array<rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr, 2> tip_pose_pubs_;
    std::array<rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr, 2> tip_force_pubs_;
    std::array<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr, 2> jac_position_joints_pubs_;
    std::array<rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr, 2> base_pose_pubs_;
    std::array<std::shared_ptr<tf2_ros::TransformBroadcaster>, 2> tip_tf_broadcasters_;
    std::array<std::shared_ptr<tf2_ros::TransformBroadcaster>, 2> base_tf_broadcasters_;

    std::array<rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr, 2> pixels_pubs_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_image_pub_;
    std::array<std::array<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr, 2>, 2> tube_marker_pubs_;
    std::array<rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr, 2> tip_position_uncertainty_marker_pubs_;
    std::array<rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr, 2> tip_force_arrow_marker_pubs_;
    std::array<rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr, 2> tip_force_uncertainty_marker_pubs_;
};