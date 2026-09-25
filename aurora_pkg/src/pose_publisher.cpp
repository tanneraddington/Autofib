#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"

#include "CombinedApi.h"
#include "PortHandleInfo.h"
#include "ToolData.h"

class AuroraPosePublisher : public rclcpp::Node
{
public:
    AuroraPosePublisher() : Node("aurora_pose_publisher")
    {
        this->declare_parameter<int>("poll_rate_hz", 30);
        this->declare_parameter<std::string>("device_port", "/dev/ttyUSB0");

        int poll_rate_hz;
        this->get_parameter("poll_rate_hz", poll_rate_hz);
        auto timer_interval = std::chrono::milliseconds(static_cast<int>(1000.0 / poll_rate_hz));

        std::string device_port;
        this->get_parameter("device_port", device_port);

        pose_pub = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("aurora/pose", 1);
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_interval),
            std::bind(&AuroraPosePublisher::timerCallback, this));

        if (capi.connect(device_port) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Connection failed to device port: %s!", device_port.c_str());
            rclcpp::shutdown();
        }

        RCLCPP_INFO(this->get_logger(), "Connected to NDI Aurora on device port: %s", device_port.c_str());

        if (capi.initialize() != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize NDICAPI!");
            rclcpp::shutdown();
        }

        RCLCPP_INFO(this->get_logger(), "NDICAPI Initialized!");
        RCLCPP_INFO(this->get_logger(), "Poll rate set to: %d Hz", poll_rate_hz);

        initializeAndEnableTools();
        capi.startTracking();
    }

    ~AuroraPosePublisher()
    {
        capi.stopTracking();
    }

private:
    void timerCallback()
    {
        std::vector<ToolData> tool_data = api_supports_BX2 ? capi.getTrackingDataBX2() : capi.getTrackingDataBX();
        
        // TODO: don't actually use multiple tools. That is not implemented yet!
        for (const auto& tool : tool_data)
        {
            publishToolData(tool);
        }
    }

    void publishToolData(const ToolData& toolData)
    {
        // TODO: make this work for more than one tool
        geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;

        pose_msg.header.stamp = this->get_clock()->now();
        pose_msg.header.frame_id = "aurora_frame";

        if (!toolData.transform.isMissing())
        {
            pose_msg.pose.pose.position.x = toolData.transform.tx;
            pose_msg.pose.pose.position.y = toolData.transform.ty;
            pose_msg.pose.pose.position.z = toolData.transform.tz;

            pose_msg.pose.pose.orientation.x = toolData.transform.qx;
            pose_msg.pose.pose.orientation.y = toolData.transform.qy;
            pose_msg.pose.pose.orientation.z = toolData.transform.qz;
            pose_msg.pose.pose.orientation.w = toolData.transform.q0;

            pose_msg.pose.covariance[0] = toolData.transform.error;

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,  // ms
                "Pose: position=(%.4f, %.4f, %.4f), orientation=(%.4f, %.4f, %.4f, %.4f), error=%.6f",
                pose_msg.pose.pose.position.x,
                pose_msg.pose.pose.position.y,
                pose_msg.pose.pose.position.z,
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w,
                pose_msg.pose.covariance[0]
            );
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Tracker tool is missing... publishing NaN pose");

            pose_msg.pose.pose.position.x = std::numeric_limits<double>::quiet_NaN();
            pose_msg.pose.pose.position.y = std::numeric_limits<double>::quiet_NaN();
            pose_msg.pose.pose.position.z = std::numeric_limits<double>::quiet_NaN();

            pose_msg.pose.pose.orientation.x = std::numeric_limits<double>::quiet_NaN();
            pose_msg.pose.pose.orientation.y = std::numeric_limits<double>::quiet_NaN();
            pose_msg.pose.pose.orientation.z = std::numeric_limits<double>::quiet_NaN();
            pose_msg.pose.pose.orientation.w = std::numeric_limits<double>::quiet_NaN();

            pose_msg.pose.covariance[0] = std::numeric_limits<double>::quiet_NaN();
        }

        pose_pub->publish(pose_msg);
    }

    void initializeAndEnableTools()
    {
        std::vector<PortHandleInfo> port_handles = capi.portHandleSearchRequest(PortHandleSearchRequestOption::NotInit);
        for (const auto& port_handle : port_handles)
        {
            capi.portHandleInitialize(port_handle.getPortHandle());
            capi.portHandleEnable(port_handle.getPortHandle());
        }
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub;
    rclcpp::TimerBase::SharedPtr timer_;
    CombinedApi capi;
    bool api_supports_BX2 = false;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::sleep_for(std::chrono::seconds(1));
    rclcpp::spin(std::make_shared<AuroraPosePublisher>());
    rclcpp::shutdown();
    return 0;
}
