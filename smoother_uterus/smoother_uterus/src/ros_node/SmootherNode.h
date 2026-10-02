
#pragma once

#include <optional>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "SmootherPipeline.h"
#include "SolutionPublisher.h"

class SmootherNode : public rclcpp::Node {
public:
    SmootherNode();

    ~SmootherNode();

private:
    void load_ros_params();
    void init_pipeline(const gtsam::Cal3_S2& camera_intrinsics);

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg);
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void left_joint_state_callback(const sensor_msgs::msg::JointState::ConstSharedPtr& msg);
    void right_joint_state_callback(const sensor_msgs::msg::JointState::ConstSharedPtr& msg);
    void camera_angle_callback(const std_msgs::msg::Float32::ConstSharedPtr& msg);

    rcl_interfaces::msg::SetParametersResult on_parameter_change(const std::vector<rclcpp::Parameter>& params);
    
    std::optional<RawSample> create_raw_sample(const sensor_msgs::msg::Image::SharedPtr& msg);
    std::optional<sensor_msgs::msg::JointState::ConstSharedPtr> lookup_left_joint_state(rclcpp::Time target_time) const;
    std::optional<sensor_msgs::msg::JointState::ConstSharedPtr> lookup_right_joint_state(rclcpp::Time target_time) const;

    // View angle in force at an image's time, and whether the view may still be moving then
    struct CameraAngleLookup {
        std::optional<double> angle_rad;
        bool settling = false;
    };
    CameraAngleLookup lookup_camera_angle(const rclcpp::Time& target_time) const;
    
    void publish_loop();

    // Setters for dynamically configurable params — own the mutation, pipeline forward, and logging.
    void set_ignore_keypoints_mode(bool value);
    void set_track_forces_mode(bool value);
    void set_camera_lag_seconds(double value);
    void set_camera_settle_seconds(double value);
    void set_left_tip_offset(const gtsam::Vector3& offset);
    void set_right_tip_offset(const gtsam::Vector3& offset);

    const int    joint_state_history_size_{250};
    const double stamp_matching_slop_seconds_{0.05};

    // Defaults live in the anonymous namespace in SmootherNode.cpp; members zero-initialised here.
    bool   track_forces_mode_{};
    bool   ignore_keypoints_mode_{};
    double camera_lag_seconds_{};
    double camera_settle_seconds_{};

    bool load_calibration_on_start_;  // if true, load calibration from the run directory
    std::string run_dir_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_joint_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_joint_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr stop_calibration_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr camera_angle_sub_;

    // View angle changes from /motor_angle (unstamped, so stamped on arrival). Only changes are stored.
    struct CameraAngleChange {
        rclcpp::Time time;
        double angle_rad;
        bool is_first;  // the first reading isn't a move: the view was already there
    };
    std::deque<CameraAngleChange> camera_angle_history_;
    std::string camera_angle_topic_;

    std::deque<sensor_msgs::msg::JointState::ConstSharedPtr> left_joint_state_history_;
    std::deque<sensor_msgs::msg::JointState::ConstSharedPtr> right_joint_state_history_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr set_params_callback_handle_;

    std::unique_ptr<SmootherPipeline> pipeline_;
    std::unique_ptr<SolutionPublisher> publisher_;
    std::thread publish_thread_;

    gtsam::Vector3 left_tip_offset_  = gtsam::Vector3::Zero();
    gtsam::Vector3 right_tip_offset_ = gtsam::Vector3::Zero();
};
