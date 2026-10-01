
#include "DataSuites.h"

#include "ros_node/SmootherNode.h"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <chrono>

using namespace std::chrono_literals;

class InitializationTests : public testing::Test {
public:
    void SetUp() override {
        rclcpp::init(0, nullptr);
        client_node = std::make_shared<rclcpp::Node>("bryant_the_client");
    }

    void TearDown() override {
        rclcpp::shutdown();
    }

    template <typename MsgT> void registerSubscriber(MsgT *targetData, const std::string &topic)
    {
        subscribers.push_back(client_node->create_subscription<MsgT>(
            topic, { 1 }, [targetData](typename MsgT::ConstSharedPtr msg) { *targetData = *msg; }));
    }

    rclcpp::Node::SharedPtr client_node;

    // maintain a collection of registered subscribers
    std::vector<rclcpp::SubscriptionBase::ConstSharedPtr> subscribers;
};

TEST_F(InitializationTests, givenSet1Input_getPoseOutput)
{
    auto node = std::make_shared<SmootherNode>();

    SingleFrameSuite1 suite{client_node};

    // publish all the necessary topics and expect a pose output
    geometry_msgs::msg::PoseWithCovarianceStamped left_base_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped right_base_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped left_tip_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped right_tip_pose;

    // load known values into the messages so we know they changed
    left_base_pose.pose.pose.position.x = 1.0;
    right_base_pose.pose.pose.position.x = 1.0;
    left_tip_pose.pose.pose.position.x = 1.0;
    right_tip_pose.pose.pose.position.x = 1.0;

    registerSubscriber(&left_base_pose, "smoother_uterus/left/base_pose");
    registerSubscriber(&right_base_pose, "smoother_uterus/right/base_pose");
    registerSubscriber(&left_tip_pose, "smoother_uterus/left/tip_pose");
    registerSubscriber(&right_tip_pose, "smoother_uterus/right/tip_pose");

    auto test_passed = [&]() {
        return (
            left_base_pose.pose.pose.position.x != 1.0 &&
            right_base_pose.pose.pose.position.x != 1.0 &&
            left_tip_pose.pose.pose.position.x != 1.0 &&
            right_tip_pose.pose.pose.position.x != 1.0
        );
    };

    // set up a 10-second timeout and publish data until we hear something
    auto timeout = rclcpp::Duration::from_seconds(10.0);
    const auto time_start = client_node->get_clock()->now();
    auto time_now = time_start;
    rclcpp::Rate publish_rate{50.0, client_node->get_clock()};

    // send it
    while (time_now < time_start + timeout && test_passed() == false) {
        suite.publishRectifiedCameraTopics();
        suite.publishJointStates();

        // ensure the callbacks for each of the nodes run
        rclcpp::spin_some(node);
        rclcpp::spin_some(client_node);

        publish_rate.sleep();
        time_now = client_node->get_clock()->now();
    }

    // check that we actually got messages
    /// @todo Check that we got close to what the poses should be
    EXPECT_NE(1.0, left_base_pose.pose.pose.position.x);
    EXPECT_NE(1.0, right_base_pose.pose.pose.position.x);
    EXPECT_NE(1.0, left_tip_pose.pose.pose.position.x);
    EXPECT_NE(1.0, right_tip_pose.pose.pose.position.x);
}

TEST_F(InitializationTests, givenSet2Input_getPoseOutput)
{
    auto node = std::make_shared<SmootherNode>();

    SingleFrameSuite2 suite{client_node};

    // publish all the necessary topics and expect a pose output
    geometry_msgs::msg::PoseWithCovarianceStamped left_base_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped right_base_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped left_tip_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped right_tip_pose;

    // load known values into the messages so we know they changed
    left_base_pose.pose.pose.position.x = 1.0;
    right_base_pose.pose.pose.position.x = 1.0;
    left_tip_pose.pose.pose.position.x = 1.0;
    right_tip_pose.pose.pose.position.x = 1.0;

    registerSubscriber(&left_base_pose, "smoother_uterus/left/base_pose");
    registerSubscriber(&right_base_pose, "smoother_uterus/right/base_pose");
    registerSubscriber(&left_tip_pose, "smoother_uterus/left/tip_pose");
    registerSubscriber(&right_tip_pose, "smoother_uterus/right/tip_pose");

    auto test_passed = [&]() {
        return (
            left_base_pose.pose.pose.position.x != 1.0 &&
            right_base_pose.pose.pose.position.x != 1.0 &&
            left_tip_pose.pose.pose.position.x != 1.0 &&
            right_tip_pose.pose.pose.position.x != 1.0
        );
    };

    // set up a 10-second timeout and publish data until we hear something
    auto timeout = rclcpp::Duration::from_seconds(10.0);
    const auto time_start = client_node->get_clock()->now();
    auto time_now = time_start;
    rclcpp::Rate publish_rate{50.0, client_node->get_clock()};

    // send it
    while (time_now < time_start + timeout && test_passed() == false) {
        suite.publishRectifiedCameraTopics();
        suite.publishJointStates();

        // ensure the callbacks for each of the nodes run
        rclcpp::spin_some(node);
        rclcpp::spin_some(client_node);

        publish_rate.sleep();
        time_now = client_node->get_clock()->now();
    }

    // check that we actually got messages
    /// @todo Check that we got close to what the poses should be
    EXPECT_NE(1.0, left_base_pose.pose.pose.position.x);
    EXPECT_NE(1.0, right_base_pose.pose.pose.position.x);
    EXPECT_NE(1.0, left_tip_pose.pose.pose.position.x);
    EXPECT_NE(1.0, right_tip_pose.pose.pose.position.x);
}

TEST_F(InitializationTests, givenSet3Input_getPoseOutput)
{
    auto node = std::make_shared<SmootherNode>();

    SingleFrameSuite3 suite{client_node};

    // publish all the necessary topics and expect a pose output
    geometry_msgs::msg::PoseWithCovarianceStamped left_base_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped right_base_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped left_tip_pose;
    geometry_msgs::msg::PoseWithCovarianceStamped right_tip_pose;

    // load known values into the messages so we know they changed
    left_base_pose.pose.pose.position.x = 1.0;
    right_base_pose.pose.pose.position.x = 1.0;
    left_tip_pose.pose.pose.position.x = 1.0;
    right_tip_pose.pose.pose.position.x = 1.0;

    registerSubscriber(&left_base_pose, "smoother_uterus/left/base_pose");
    registerSubscriber(&right_base_pose, "smoother_uterus/right/base_pose");
    registerSubscriber(&left_tip_pose, "smoother_uterus/left/tip_pose");
    registerSubscriber(&right_tip_pose, "smoother_uterus/right/tip_pose");

    auto test_passed = [&]() {
        return (
            left_base_pose.pose.pose.position.x != 1.0 &&
            right_base_pose.pose.pose.position.x != 1.0 &&
            left_tip_pose.pose.pose.position.x != 1.0 &&
            right_tip_pose.pose.pose.position.x != 1.0
        );
    };

    // set up a 10-second timeout and publish data until we hear something
    auto timeout = rclcpp::Duration::from_seconds(10.0);
    const auto time_start = client_node->get_clock()->now();
    auto time_now = time_start;
    rclcpp::Rate publish_rate{50.0, client_node->get_clock()};

    // send it
    while (time_now < time_start + timeout && test_passed() == false) {
        suite.publishRectifiedCameraTopics();
        suite.publishJointStates();

        // ensure the callbacks for each of the nodes run
        rclcpp::spin_some(node);
        rclcpp::spin_some(client_node);

        publish_rate.sleep();
        time_now = client_node->get_clock()->now();
    }

    // check that we actually got messages
    /// @todo Check that we got close to what the poses should be
    EXPECT_NE(1.0, left_base_pose.pose.pose.position.x);
    EXPECT_NE(1.0, right_base_pose.pose.pose.position.x);
    EXPECT_NE(1.0, left_tip_pose.pose.pose.position.x);
    EXPECT_NE(1.0, right_tip_pose.pose.pose.position.x);
}
