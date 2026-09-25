
#include "ros_node/SmootherNode.h"

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <chrono>

using namespace std::chrono_literals;

class SmootherTfTests : public testing::Test {
public:
    void SetUp() override {
        rclcpp::init(0, nullptr);
        client_node = std::make_shared<rclcpp::Node>("bryant_the_client");
        client_node->set_parameter({"use_sim_time", true});

        /// @todo Install a bogus calibration file and expect smoother to broadcast the transforms on startup
        client_node->declare_parameter("calibration_file_load_path", "/calibration/calibration.yaml");
        client_node->declare_parameter("calibration_file_save_path", "/calibration/calibration.yaml");

        clock_publisher = client_node->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
    }

    void TearDown() override {
        clock_publisher = nullptr;

        rclcpp::shutdown();
    }

    std::shared_ptr<SmootherNode> createNode() {
        auto node = std::make_shared<SmootherNode>();
        node->set_parameter({"use_sim_time", true});
        return node;
    }

    void advanceTime(std::shared_ptr<SmootherNode> node,
                     const std::chrono::microseconds& duration = {100ms}) {
        // increment the clock
        now += duration;

        // publish the new time
        rosgraph_msgs::msg::Clock msg;
        msg.clock = now;
        clock_publisher->publish(msg);

        // trigger callbacks in each node
        // node->cycle();
        rclcpp::spin_some(node);
        rclcpp::spin_some(client_node);
    }

    using ClockPublisherPtr = std::shared_ptr<rclcpp::Publisher<rosgraph_msgs::msg::Clock>>;

    rclcpp::Node::SharedPtr client_node;
    rclcpp::Time now {1, 0};
    ClockPublisherPtr clock_publisher;

};

void printTransform(const geometry_msgs::msg::TransformStamped& transform)
{
    std::cout << "Transform: " << transform.header.frame_id << " to " << transform.child_frame_id
              << " at (" << transform.header.stamp.sec << ", " << transform.header.stamp.nanosec << ")" << std::endl;
    std::cout << "Translation: (" << transform.transform.translation.x << ", "
              << transform.transform.translation.y << ", "
              << transform.transform.translation.z << ")" << std::endl;
    std::cout << "Rotation: (" << transform.transform.rotation.w << ", "
              << transform.transform.rotation.x << ", "
              << transform.transform.rotation.y << ", "
              << transform.transform.rotation.z << ")" << std::endl;
}

TEST_F(SmootherTfTests, givenInit_broadcastInitialTransforms)
{
    GTEST_SKIP() << "The TF requirement for ves_smoother has been obsoleted by aliss_core's launch file";

    auto node = createNode();

    auto tf_buffer = std::make_unique<tf2_ros::Buffer>(client_node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    // give TF some time to percolate
    advanceTime(node, 500ms);

    try {
        // expect TF connections:
        // endoscope_optical ---> ves_smoother/camera (identity transform)
        // ves_smoother/camera ---> ves_smoother/left/base
        // ves_smoother/camera ---> ves_smoother/right/base
        auto transform1 = tf_buffer->lookupTransform("endoscope_optical", "ves_smoother/camera", now);
        auto transform2 = tf_buffer->lookupTransform("ves_smoother/camera", "ves_smoother/left/base", now);
        auto transform3 = tf_buffer->lookupTransform("ves_smoother/camera", "ves_smoother/right/base", now);

        printTransform(transform1);
        printTransform(transform2);
        printTransform(transform3);

        EXPECT_NEAR(transform1.transform.translation.x, 0.0, 1e-6);
        EXPECT_NEAR(transform1.transform.translation.y, 0.0, 1e-6);
        EXPECT_NEAR(transform1.transform.translation.z, 0.0, 1e-6);
        EXPECT_NEAR(transform1.transform.rotation.w, 1.0, 1e-6);
        EXPECT_NEAR(transform1.transform.rotation.x, 0.0, 1e-6);
        EXPECT_NEAR(transform1.transform.rotation.y, 0.0, 1e-6);
        EXPECT_NEAR(transform1.transform.rotation.z, 0.0, 1e-6);

        EXPECT_NE(transform2.transform.translation.x, 0.0);
        EXPECT_NE(transform3.transform.translation.x, 0.0);
    } catch (const tf2::LookupException& e) {
        FAIL() << "TF lookup failed: " << e.what() << std::endl;
    }
}
