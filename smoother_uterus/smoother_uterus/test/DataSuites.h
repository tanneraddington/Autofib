/**
 * Helper class to create data suites, i.e. single- or multi-frame ROS-facing data.
 * You can add more or less data to a single frame, but your unit test is responsible for
 * handling/testing the data properly.
 *
 * @author joshpetrin
 */

#pragma once

#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_geometry/pinhole_camera_model.hpp>
#include <image_transport/camera_publisher.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <filesystem>
#include <string>

static const std::filesystem::path& getAssetDirectory() {
    const static auto path =
        std::filesystem::current_path() / "test_assets";
    return path;
}

class SmootherSingleFrameSuite {
public:
    SmootherSingleFrameSuite(
        rclcpp::Node::SharedPtr node,
        std::filesystem::path image_path,
        std::filesystem::path camera_info_path
    )
    : node_{*node}
    , image_transport_{node}
    , info_manager_{node.get(), "ves_camera"}
    , camera_model_{}
    , raw_image_{cv::imread(image_path)}
    , rectified_image_{}
    , raw_camera_pub_{image_transport_.advertiseCamera("camera/image", 1)}
    , rect_camera_pub_{image_transport_.advertiseCamera("camera/image_rect", 1)}
    , left_joint_pub_{
        node->create_publisher<sensor_msgs::msg::JointState>("robot/left/joint/measured_jp", 1)}
    , right_joint_pub_{
        node->create_publisher<sensor_msgs::msg::JointState>("robot/right/joint/measured_jp", 1)}
    {
        info_manager_.loadCameraInfo("file://" + std::string{camera_info_path});
        camera_model_.fromCameraInfo(info_manager_.getCameraInfo());
        camera_model_.rectifyImage(raw_image_, rectified_image_);

        RCLCPP_INFO_STREAM(
            node_.get_logger(), "Raw camera image topic: " << raw_camera_pub_.getTopic());
        RCLCPP_INFO_STREAM(
            node_.get_logger(), "Raw camera info topic: " << raw_camera_pub_.getInfoTopic()); 
        RCLCPP_INFO_STREAM(
            node_.get_logger(), "Rectified camera image topic: " << rect_camera_pub_.getTopic());
        RCLCPP_INFO_STREAM(
            node_.get_logger(), "Rectified camera info topic: " << rect_camera_pub_.getInfoTopic()); 
    }

    void publishRawCameraTopics() {
        std_msgs::msg::Header header;
        // fudge the camera timestamp to circumvent the lag factor
        const rclcpp::Duration fudge {std::chrono::milliseconds{70}};
        header.stamp = node_.get_clock()->now() + fudge;

        cv_bridge::CvImage cv_image{header, "bgr8", raw_image_};
        sensor_msgs::msg::Image image_msg;
        cv_image.toImageMsg(image_msg);
        image_msg.header = header;
        raw_camera_pub_.publish(image_msg, info_manager_.getCameraInfo());
    }

    void publishRectifiedCameraTopics() {
        std_msgs::msg::Header header;
        // fudge the camera timestamp to circumvent the lag factor
        const rclcpp::Duration fudge {std::chrono::milliseconds{70}};
        header.stamp = node_.get_clock()->now() + fudge;

        cv_bridge::CvImage cv_image{header, "bgr8", rectified_image_};
        sensor_msgs::msg::Image image_msg;
        cv_image.toImageMsg(image_msg);
        image_msg.header = header;
        rect_camera_pub_.publish(image_msg, info_manager_.getCameraInfo());
    }

    void publishJointStates() {
        std_msgs::msg::Header header;
        header.stamp = node_.get_clock()->now();

        left_filt_measured_joint_msg_.header = header;
        right_filt_measured_joint_msg_.header = header;
        left_joint_pub_->publish(left_filt_measured_joint_msg_);
        right_joint_pub_->publish(right_filt_measured_joint_msg_);
    }

protected:
    void setLeftFilteredMeasuredJointMsg(const sensor_msgs::msg::JointState& msg) {
        left_filt_measured_joint_msg_ = msg;
    }
    void setRightFilteredMeasuredJointMsg(const sensor_msgs::msg::JointState& msg) {
        right_filt_measured_joint_msg_ = msg;
    }

    static std::vector<std::string> getJointNames() {
        return {"inner_rotation",
                "outer_rotation",
                "inner_translation",
                "outer_translation",
                "tool"};
    }

private:
    rclcpp::Node& node_;

    image_transport::ImageTransport image_transport_;
    camera_info_manager::CameraInfoManager info_manager_;

    image_geometry::PinholeCameraModel camera_model_;
    cv::Mat raw_image_;
    cv::Mat rectified_image_;

    sensor_msgs::msg::JointState left_filt_measured_joint_msg_;
    sensor_msgs::msg::JointState right_filt_measured_joint_msg_;

    image_transport::CameraPublisher raw_camera_pub_;
    image_transport::CameraPublisher rect_camera_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_joint_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr right_joint_pub_;
};

class SingleFrameSuite1 : public SmootherSingleFrameSuite {
public:
    SingleFrameSuite1(rclcpp::Node::SharedPtr node)
    : SmootherSingleFrameSuite{
        node,
        getAssetDirectory() / "frame_1.png",
        getAssetDirectory() / "camera_calibration.yaml"}
    {
        sensor_msgs::msg::JointState left_joint_state;
        sensor_msgs::msg::JointState right_joint_state;

        left_joint_state.name = getJointNames();
        right_joint_state.name = getJointNames();

        left_joint_state.position = {3.71526228, 0.731938426, 0.0273545434, 0.00332526975};
        right_joint_state.position = {3.30079353, 6.18345579, 0.020893874, 0.000001};

        setLeftFilteredMeasuredJointMsg(left_joint_state);
        setRightFilteredMeasuredJointMsg(right_joint_state);
    }
};

class SingleFrameSuite2 : public SmootherSingleFrameSuite {
public:
    SingleFrameSuite2(rclcpp::Node::SharedPtr node)
    : SmootherSingleFrameSuite{
        node,
        getAssetDirectory() / "frame_2.png",
        getAssetDirectory() / "camera_calibration.yaml"}
    {
        sensor_msgs::msg::JointState left_joint_state;
        sensor_msgs::msg::JointState right_joint_state;

        left_joint_state.name = getJointNames();
        right_joint_state.name = getJointNames();

        left_joint_state.position = {3.43616940, 0.105454368, 0.0239505608, 0.001168038};
        right_joint_state.position = {3.13732620, 1.51455879, 0.0224374762, 0.002258535};

        setLeftFilteredMeasuredJointMsg(left_joint_state);
        setRightFilteredMeasuredJointMsg(right_joint_state);
    }
};

class SingleFrameSuite3 : public SmootherSingleFrameSuite {
public:
    SingleFrameSuite3(rclcpp::Node::SharedPtr node)
    : SmootherSingleFrameSuite{
        node,
        getAssetDirectory() / "frame_3.png",
        getAssetDirectory() / "camera_calibration.yaml"}
    {
        sensor_msgs::msg::JointState left_joint_state;
        sensor_msgs::msg::JointState right_joint_state;

        left_joint_state.name = getJointNames();
        right_joint_state.name = getJointNames();

        left_joint_state.position = {4.12436203, 2.54912814, 0.018878462, 0.000978456};
        right_joint_state.position = {3.13723033, 6.17667051, 0.03521355, 0.00280896};

        setLeftFilteredMeasuredJointMsg(left_joint_state);
        setRightFilteredMeasuredJointMsg(right_joint_state);
    }
};
