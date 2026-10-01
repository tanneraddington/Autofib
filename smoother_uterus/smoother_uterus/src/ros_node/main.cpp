#include <rclcpp/rclcpp.hpp>

#include "SmootherNode.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    // Standard single threaded node to process callbacks and publish results.
    // Heavy lifting done in SmootherPipeline on separate threads.
    auto node = std::make_shared<SmootherNode>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}