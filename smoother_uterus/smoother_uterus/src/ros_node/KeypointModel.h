#pragma once

#include <onnxruntime_cxx_api.h>
#include <gtsam/geometry/Point2.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

struct KeypointInferenceResult {
    std::vector<gtsam::Point2> left_keypoints;
    std::vector<float> left_scores;
    std::vector<gtsam::Point2> right_keypoints;
    std::vector<float> right_scores;
    double inference_time_ms = 0.0;
};

class KeypointModel {
public:
    KeypointModel(const rclcpp::Logger& logger, const std::string& model_path);

    KeypointInferenceResult process_image(const cv::Mat& image);

private:
    std::vector<Ort::Value> run_inference(
        const cv::Mat& input_tensor_values,
        const std::array<int64_t, 4>& input_shape);

    cv::Mat preprocess_image(const cv::Mat& image) const;

    KeypointInferenceResult parse_outputs(
        const std::vector<Ort::Value>& output_tensors,
        const cv::Size& image_size) const;

    rclcpp::Logger logger_;
    Ort::Env onnx_env_{ORT_LOGGING_LEVEL_WARNING, "keypoint_model"};
    std::unique_ptr<Ort::Session> onnx_session_;
    Ort::MemoryInfo memory_info_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Storage for the c_str() pointers above — must outlive the pointer arrays
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
};
