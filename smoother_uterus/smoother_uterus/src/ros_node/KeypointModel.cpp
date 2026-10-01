#include "KeypointModel.h"

#include <chrono>

namespace {
    constexpr int    MODEL_INPUT_SIZE       = 256;
    constexpr float  SCORE_THRESHOLD        = 0.8f;
    constexpr size_t LABELS_OUTPUT          = 1;
    constexpr size_t SCORES_OUTPUT          = 2;
    constexpr size_t KEYPOINTS_OUTPUT       = 3;
    constexpr int    LEFT_LABEL             = 1;
    constexpr int    RIGHT_LABEL            = 2;
}

KeypointModel::KeypointModel(const rclcpp::Logger& logger, const std::string& model_path)
:
    logger_(logger),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;  // Reasonable assumption for most systems with a single GPU
        // The default (exhaustive) benchmarks every conv algorithm each time a new input shape shows up,
        // and the keypoint head's shape changes with the number of detections, so it keeps re-tuning.
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        options.AppendExecutionProvider_CUDA(cuda_options);
    }
    catch (const Ort::Exception& e) {
        RCLCPP_ERROR(logger_, "CUDA unavailable, falling back to CPU (will be VERY slow): %s", e.what());
    }

    onnx_session_ = std::make_unique<Ort::Session>(onnx_env_, model_path.c_str(), options);

    // Pre-fetch input and output names to avoid doing this on every inference call
    Ort::AllocatorWithDefaultOptions allocator;

    // We have to do this in two loops each to ensure the storage exists
    for (size_t i = 0; i < onnx_session_->GetInputCount(); ++i) {
        auto name = onnx_session_->GetInputNameAllocated(i, allocator);
        input_name_storage_.emplace_back(name.get());
    }
    for (const auto& s : input_name_storage_) {
        input_names_.push_back(s.c_str());
    }
    for (size_t i = 0; i < onnx_session_->GetOutputCount(); ++i) {
        auto name = onnx_session_->GetOutputNameAllocated(i, allocator);
        output_name_storage_.emplace_back(name.get());
    }
    for (const auto& s : output_name_storage_) {
        output_names_.push_back(s.c_str());
    }

    // Warm up once here: the first CUDA run sets up kernels and memory and can take seconds
    try {
        auto t0 = std::chrono::steady_clock::now();
        process_image(cv::Mat(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, CV_8UC3, cv::Scalar(0, 0, 0)));
        RCLCPP_INFO(logger_, "Keypoint model warm-up took %.0f ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
    }
    catch (const std::exception& e) {
        RCLCPP_WARN(logger_, "Keypoint model warm-up failed: %s", e.what());
    }
}

KeypointInferenceResult KeypointModel::process_image(const cv::Mat& image)
{
    // Any input size is resized to MODEL_INPUT_SIZE; keypoints are scaled back per axis in parse_outputs.
    // (The Virtuoso model was trained on 1080x1080 frames; retrain for the uterine robot's camera.)
    if (image.empty())
        return {};

    auto t0 = std::chrono::steady_clock::now();

    // Convert image to input blob that ONNX expects
    cv::Mat input_tensor = cv::dnn::blobFromImage(
        image,
        1.0 / 255.0,                                  // Normalize scale factor
        cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), // resize to model input size
        cv::Scalar(),                                 // no mean subtraction
        true,                                         // BGR to RGB
        false,                                        // no crop
        CV_32F                                        // output type
    );

    if (input_tensor.empty())
        return {};

    auto outputs = run_inference(input_tensor, {1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE});

    auto result = parse_outputs(outputs, image.size());
    result.inference_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return result;
}

std::vector<Ort::Value>
KeypointModel::run_inference(const cv::Mat& input_tensor_values, const std::array<int64_t, 4>& input_shape)
{
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        reinterpret_cast<float*>(input_tensor_values.data),
        input_tensor_values.total(),
        input_shape.data(),
        input_shape.size());

    return onnx_session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        input_names_.size(),
        output_names_.data(),
        output_names_.size());
}

KeypointInferenceResult KeypointModel::parse_outputs(
    const std::vector<Ort::Value>& outputs,
    const cv::Size& image_size) const
{
    // Get the labels, scores, and keypoints from the output tensors
    const auto* labels = outputs[LABELS_OUTPUT].GetTensorData<int64_t>();
    const auto* scores = outputs[SCORES_OUTPUT].GetTensorData<float>();
    const auto* keypoints = outputs[KEYPOINTS_OUTPUT].GetTensorData<float>();

    auto shape = outputs[KEYPOINTS_OUTPUT].GetTensorTypeAndShapeInfo().GetShape();

    const int64_t num_dets = shape[0];
    const int64_t K        = shape[1];
    const int64_t D        = shape[2];

    const float scale_x = static_cast<float>(image_size.width)  / static_cast<float>(MODEL_INPUT_SIZE);
    const float scale_y = static_cast<float>(image_size.height) / static_cast<float>(MODEL_INPUT_SIZE);

    KeypointInferenceResult result;

    for (int64_t i = 0; i < num_dets; ++i) {
        const float score = scores[i];

        // Filter out low-confidence detections
        if (score < SCORE_THRESHOLD)
            continue;

        const int label = static_cast<int>(labels[i]);

        const int64_t base = i * K * D;
        const float x = keypoints[base + 0] * scale_x;
        const float y = keypoints[base + 1] * scale_y;

        auto point = gtsam::Point2(x, y);

        if (label == LEFT_LABEL) {
            result.left_keypoints.push_back(point);
            result.left_scores.push_back(score);
        }
        else if (label == RIGHT_LABEL) {
            result.right_keypoints.push_back(point);
            result.right_scores.push_back(score);
        }
    }

    return result;
}