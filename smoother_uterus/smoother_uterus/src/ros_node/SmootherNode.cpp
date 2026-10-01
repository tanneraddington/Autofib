#include "SmootherNode.h"

#include <optional>
#include <string>
#include <filesystem>
#include <cmath>

#include <rclcpp/logger.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <Eigen/Dense>

#include "CalibrationIo.h"

using namespace gtsam;

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::JointState;
using std_msgs::msg::Empty;

namespace {
    // Default ROS parameter values
    constexpr bool   DEFAULT_IGNORE_KEYPOINTS_MODE = false;
    constexpr bool   DEFAULT_TRACK_FORCES_MODE     = false;
    constexpr double DEFAULT_CAMERA_LAG_SECONDS    = 0.07;   // lag between image timestamps and actual capture time
    constexpr double DEFAULT_LEFT_TIP_OFFSET_Z     = 0.004;  // spatula pusher
    constexpr double DEFAULT_RIGHT_TIP_OFFSET_Z    = 0.008;  // cautery probe

    // Default topics for the uterine robot (override with ROS params of the same name).
    // Joint states come from motor_node: [inner_rot, outer_rot, inner_trans, outer_trans] in rad / m.
    // Images must be RECTIFIED, with a matching CameraInfo (the robot's raw feed is /image).
    constexpr const char* DEFAULT_LEFT_JOINT_STATE_TOPIC   = "/robot/left/joint/measured_jp";
    constexpr const char* DEFAULT_RIGHT_JOINT_STATE_TOPIC  = "/robot/right/joint/measured_jp";
    constexpr const char* DEFAULT_CAMERA_JOINT_STATE_TOPIC = "/robot/camera/joint/measured_jp";
    constexpr const char* DEFAULT_IMAGE_TOPIC              = "/camera/image_rect";
    constexpr const char* DEFAULT_CAMERA_INFO_TOPIC        = "/camera/camera_info";

    // Camera motor motion beyond this (from the first reading) invalidates the fixed-camera model
    constexpr double CAMERA_MOTION_WARN_RAD = 1.0 * M_PI / 180.0;

    // Constants (could be ROS params later if we want to be fancy)
    constexpr double SMALL_FORCE_STD = 1e-2; // Near-zero force prior when not estimating forces; squared gives 1e-4 variance (matching the old 0.0001 * I covariance)
    constexpr double FORCE_PRIOR_STD = 1.0; // Force prior uncertainty when in track forces mode; set to a reasonable value to allow non-zero forces without being too noisy

    // Performance tracking window
    struct PerfSample {
        std::chrono::steady_clock::time_point timestamp;
        double inference_time_ms;
        double tracking_time_ms;
    };
    constexpr double PERF_WINDOW_SECONDS = 5.0;
}

SmootherNode::SmootherNode() : Node{"smoother_uterus"}
{
    load_ros_params();

    // All publishing handled by solution publisher
    publisher_ = std::make_unique<SolutionPublisher>(*this);

    // Topic names (static, read once at startup)
    const auto left_joint_state_topic   = this->declare_parameter<std::string>("left_joint_state_topic",   DEFAULT_LEFT_JOINT_STATE_TOPIC);
    const auto right_joint_state_topic  = this->declare_parameter<std::string>("right_joint_state_topic",  DEFAULT_RIGHT_JOINT_STATE_TOPIC);
    const auto camera_joint_state_topic = this->declare_parameter<std::string>("camera_joint_state_topic", DEFAULT_CAMERA_JOINT_STATE_TOPIC);
    const auto image_topic              = this->declare_parameter<std::string>("image_topic",              DEFAULT_IMAGE_TOPIC);
    const auto camera_info_topic        = this->declare_parameter<std::string>("camera_info_topic",        DEFAULT_CAMERA_INFO_TOPIC);

    RCLCPP_INFO(this->get_logger(), "Joint states: %s, %s | image: %s | camera_info: %s | camera joint: %s",
        left_joint_state_topic.c_str(), right_joint_state_topic.c_str(),
        image_topic.c_str(), camera_info_topic.c_str(), camera_joint_state_topic.c_str());

    // Subscribe to left and right joint states
    left_joint_state_sub_ = this->create_subscription<JointState>(
        left_joint_state_topic, 10,
        std::bind(&SmootherNode::left_joint_state_callback, this, std::placeholders::_1));

    right_joint_state_sub_ = this->create_subscription<JointState>(
        right_joint_state_topic, 10,
        std::bind(&SmootherNode::right_joint_state_callback, this, std::placeholders::_1));

    // Camera motor angle, only used to check that the camera stays put
    camera_joint_state_sub_ = this->create_subscription<JointState>(
        camera_joint_state_topic, 10,
        std::bind(&SmootherNode::camera_joint_state_callback, this, std::placeholders::_1));

    // Subscribe to camera info and image
    camera_info_sub_ = this->create_subscription<CameraInfo>(
        camera_info_topic, 10,
        std::bind(&SmootherNode::camera_info_callback, this, std::placeholders::_1));

    image_sub_ = this->create_subscription<Image>(
        image_topic, 1, // only process most recent image
        std::bind(&SmootherNode::image_callback, this, std::placeholders::_1));

    // Subscription to stop calibration signal, sets flag to lock in current calibration and stop updating with new data
    stop_calibration_sub_ = this->create_subscription<Empty>(
        "/smoother_uterus/stop_calibration", 10,
        [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
            if (!pipeline_) {
                RCLCPP_WARN(this->get_logger(), "Received stop calibration signal but pipeline is not initialized yet.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Received stop calibration signal, stopping calibration.");
            pipeline_->stop_calibration();
        });

    set_params_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&SmootherNode::on_parameter_change, this, std::placeholders::_1));
}

SmootherNode::~SmootherNode()
{
    if (publish_thread_.joinable()) {
        publish_thread_.join();
    }
}

void SmootherNode::init_pipeline(const Cal3_S2& camera_intrinsics)
{
    // Load calibration file if requested, else start in calibrating mode
    std::optional<SmootherCalibration> calibration = std::nullopt;

    if (load_calibration_on_start_)
    {
        std::string calibration_file = run_dir_ + "/calibration/smoother/smoother_calibration.yaml";
        RCLCPP_INFO(this->get_logger(), "Loading calibration from: %s", calibration_file.c_str());

        LoadedCalibration loaded_calibration = load_calibration_yaml(calibration_file);
        calibration = loaded_calibration.calibration;

        double days_old = loaded_calibration.age_seconds / (60.0 * 60.0 * 24.0);
        RCLCPP_INFO(this->get_logger(), "Loaded calibration from file. Calibration is %.2f days old.", days_old);
    }

    std::filesystem::path share_directory = ament_index_cpp::get_package_share_directory("smoother_uterus");
    std::filesystem::path keypoint_model_path = share_directory / "models/keypoint_rcnn.onnx";

    pipeline_ = std::make_unique<SmootherPipeline>(
        this->get_logger(),
        camera_intrinsics,
        keypoint_model_path.string(),
        calibration);

    pipeline_->set_tip_offsets(left_tip_offset_, right_tip_offset_);
    RCLCPP_INFO(this->get_logger(), "Pipeline initialized.");

    // Start the publishing thread, which will continuously check for new solutions to publish
    publish_thread_ = std::thread(&SmootherNode::publish_loop, this);
    RCLCPP_INFO(this->get_logger(), "Publishing thread started.");
}

void SmootherNode::camera_info_callback(const CameraInfo::SharedPtr camera_info_msg)
{
    Cal3_S2 camera_intrinsics(
        camera_info_msg->p[0],  // fx
        camera_info_msg->p[5],  // fy
        0.0,                    // skew (rectified)
        camera_info_msg->p[2],  // cx
        camera_info_msg->p[6]   // cy
    );

    RCLCPP_INFO(this->get_logger(), "Received camera info. Initializing pipeline with camera intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
        camera_intrinsics.fx(), camera_intrinsics.fy(), camera_intrinsics.px(), camera_intrinsics.py());

    // Now that we finally have the camera intrinsics, we can initialize the estimation pipeline
    init_pipeline(camera_intrinsics);
    RCLCPP_INFO(this->get_logger(), "Pipeline initialized and ready to process raw data samples.");

    // Unsubscribe after first message since we only need camera info to init pipeline once
    camera_info_sub_.reset();
}

static void push_joint_state_bounded(
    std::deque<JointState::ConstSharedPtr>& history,
    const JointState::ConstSharedPtr& msg,
    size_t max_size)
{
    history.emplace_back(msg);
    if (history.size() > max_size)
        history.pop_front();
}

void SmootherNode::left_joint_state_callback(const JointState::ConstSharedPtr& msg)
{
    push_joint_state_bounded(left_joint_state_history_, msg, joint_state_history_size_);
}

void SmootherNode::right_joint_state_callback(const JointState::ConstSharedPtr& msg)
{
    push_joint_state_bounded(right_joint_state_history_, msg, joint_state_history_size_);
}

void SmootherNode::camera_joint_state_callback(const JointState::ConstSharedPtr& msg)
{
    if (msg->position.empty())
        return;

    const double angle = msg->position[0];
    if (!reference_camera_angle_) {
        reference_camera_angle_ = angle;
        RCLCPP_INFO(this->get_logger(), "Camera motor angle: %.3f rad (must stay fixed while tracking)", angle);
        return;
    }

    if (std::abs(angle - *reference_camera_angle_) > CAMERA_MOTION_WARN_RAD) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Camera motor moved %.1f deg from its starting angle. The smoother assumes a fixed camera, so "
            "calibration and tracking are invalid while it moves (run motor_node with camera_tracking:=false).",
            (angle - *reference_camera_angle_) * 180.0 / M_PI);
    }
}

std::optional<JointState::ConstSharedPtr> lookup_joint_state(
    const std::deque<JointState::ConstSharedPtr>& history, rclcpp::Time target_time, double slop_seconds)
{
    // Return if we dont have any joint states yet
    if (history.empty()) {
        return std::nullopt;
    }

    // Single-threaded executor: all callbacks (including joint state) run on the same thread as this call, so no lock needed.
    auto best_it = history.begin();

    double best_diff = std::abs((target_time - (*best_it)->header.stamp).seconds());

    for (auto it = history.begin(); it != history.end(); ++it) {
        double diff = std::abs((target_time - (*it)->header.stamp).seconds());
        if (diff < best_diff) {
            best_diff = diff;
            best_it = it;
        }
    }

    // If the best match is outside the slop window, return nullopt (failure)
    if (best_diff > slop_seconds) {
        return std::nullopt;
    }

    return *best_it;
}

std::optional<JointState::ConstSharedPtr> SmootherNode::lookup_left_joint_state(rclcpp::Time target_time) const
{
    return lookup_joint_state(left_joint_state_history_, target_time, stamp_matching_slop_seconds_);
}

std::optional<JointState::ConstSharedPtr> SmootherNode::lookup_right_joint_state(rclcpp::Time target_time) const
{
    return lookup_joint_state(right_joint_state_history_, target_time, stamp_matching_slop_seconds_);
}

std::optional<RawSample> SmootherNode::create_raw_sample(const Image::SharedPtr& msg)
{
    // Clone to give the cv::Mat ownership of its data — toCvShare wraps the ROS
    // message buffer without owning it, which would become a dangling reference
    // once image_callback returns and the message shared_ptr goes out of scope.
    auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    cv::Mat image = cv_ptr->image.clone();

    // Adjust image timestamp to account for camera lag
    rclcpp::Time stamp = rclcpp::Time(msg->header.stamp) - rclcpp::Duration::from_seconds(camera_lag_seconds_);

    // Use adjusted stamp to lookup joint states
    std::optional<JointState::ConstSharedPtr> left_joint_state = lookup_left_joint_state(stamp);
    std::optional<JointState::ConstSharedPtr> right_joint_state = lookup_right_joint_state(stamp);

    // Return if we can't find matching joint states for the image
    if (!left_joint_state || !right_joint_state) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Image callback: Received image but could not find matching joint states."
            "Are joint states being published?");
        return std::nullopt;
    }

    // Expect [inner_rot, outer_rot, inner_trans, outer_trans]
    if ((*left_joint_state)->position.size() < 4 || (*right_joint_state)->position.size() < 4) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Image callback: joint states need 4 positions [inner_rot, outer_rot, inner_trans, outer_trans].");
        return std::nullopt;
    }

    // Force covariance for the force prior
    Vector3Gaussian force_prior;
    force_prior.mean = Vector3::Zero();
    if (track_forces_mode_) {
        force_prior.cov = FORCE_PRIOR_STD * FORCE_PRIOR_STD * Matrix3::Identity();
    } else {
        force_prior.cov = SMALL_FORCE_STD * SMALL_FORCE_STD * Matrix3::Identity(); // near-zero prior when not estimating forces
    }

    // Push new sample onto pipeline
    RawSample sample{
        .stamp = stamp,
        .image = image,
        .left_joint_values = gtsam::Vector4(
            (*left_joint_state)->position[0],
            (*left_joint_state)->position[1],
            (*left_joint_state)->position[2],
            (*left_joint_state)->position[3]),
        .right_joint_values = gtsam::Vector4(
            (*right_joint_state)->position[0],
            (*right_joint_state)->position[1],
            (*right_joint_state)->position[2],
            (*right_joint_state)->position[3]),
        .left_tip_force_prior = force_prior,
        .right_tip_force_prior = force_prior,
        .ignore_keypoints = ignore_keypoints_mode_
    };

    return sample;
}

void SmootherNode::image_callback(const Image::SharedPtr msg)
{
    // Check if pipeline is initialized, if not we can't do anything with the image so just return
    if (!pipeline_) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Received image but pipeline is not initialized yet. Are camera_info messages being published?");
        return;
    }

    // This starts the full pipeline using CVs to pass data between threads and publish to ROS topics
    auto sample = create_raw_sample(msg);
    if (sample) {
        pipeline_->push_raw_sample(*sample);
    } else {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Received image but failed to create raw sample. Are joint states being published?");
    }
}

void SmootherNode::set_ignore_keypoints_mode(bool value)
{
    ignore_keypoints_mode_ = value;
    RCLCPP_INFO(this->get_logger(), "ignore_keypoints_mode: %s", value ? "true" : "false");
}

void SmootherNode::set_track_forces_mode(bool value)
{
    track_forces_mode_ = value;
    RCLCPP_INFO(this->get_logger(), "track_forces_mode: %s", value ? "true" : "false");
}

void SmootherNode::set_camera_lag_seconds(double value)
{
    camera_lag_seconds_ = value;
    RCLCPP_INFO(this->get_logger(), "camera_lag_seconds: %.3f s", value);
}

void SmootherNode::set_left_tip_offset(const gtsam::Vector3& offset)
{
    left_tip_offset_ = offset;
    if (pipeline_) pipeline_->set_tip_offsets(left_tip_offset_, right_tip_offset_);
    RCLCPP_INFO(this->get_logger(), "left_tip_offset: [%.5f, %.5f, %.5f]",
        offset.x(), offset.y(), offset.z());
}

void SmootherNode::set_right_tip_offset(const gtsam::Vector3& offset)
{
    right_tip_offset_ = offset;
    if (pipeline_) pipeline_->set_tip_offsets(left_tip_offset_, right_tip_offset_);
    RCLCPP_INFO(this->get_logger(), "right_tip_offset: [%.5f, %.5f, %.5f]",
        offset.x(), offset.y(), offset.z());
}

void SmootherNode::load_ros_params()
{
    load_calibration_on_start_ = this->declare_parameter<bool>("load", false);
    run_dir_ = this->declare_parameter<std::string>("run_dir", "");

    set_ignore_keypoints_mode(this->declare_parameter<bool>("ignore_keypoints_mode", DEFAULT_IGNORE_KEYPOINTS_MODE));
    set_track_forces_mode    (this->declare_parameter<bool>("track_forces_mode",     DEFAULT_TRACK_FORCES_MODE));
    set_camera_lag_seconds   (this->declare_parameter<double>("camera_lag_seconds",  DEFAULT_CAMERA_LAG_SECONDS));

    set_left_tip_offset({
        this->declare_parameter<double>("left.tip_offset.x", 0.0),
        this->declare_parameter<double>("left.tip_offset.y", 0.0),
        this->declare_parameter<double>("left.tip_offset.z", DEFAULT_LEFT_TIP_OFFSET_Z),
    });
    set_right_tip_offset({
        this->declare_parameter<double>("right.tip_offset.x", 0.0),
        this->declare_parameter<double>("right.tip_offset.y", 0.0),
        this->declare_parameter<double>("right.tip_offset.z", DEFAULT_RIGHT_TIP_OFFSET_Z),
    });
}

rcl_interfaces::msg::SetParametersResult SmootherNode::on_parameter_change(
    const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    auto new_left_offset = left_tip_offset_;
    auto new_right_offset = right_tip_offset_;
    bool left_offset_changed = false;
    bool right_offset_changed = false;

    for (const auto& param : params) {
        const std::string& name = param.get_name();

        if (name == "ignore_keypoints_mode") {
            set_ignore_keypoints_mode(param.as_bool());
        }
        else if (name == "track_forces_mode") {
            set_track_forces_mode(param.as_bool());
        }
        else if (name == "camera_lag_seconds") {
            set_camera_lag_seconds(param.as_double());
        }
        else if (name == "left.tip_offset.x") {
            new_left_offset.x() = param.as_double(); left_offset_changed = true;
        }
        else if (name == "left.tip_offset.y") {
            new_left_offset.y() = param.as_double(); left_offset_changed = true;
        }
        else if (name == "left.tip_offset.z") {
            new_left_offset.z() = param.as_double(); left_offset_changed = true;
        }
        else if (name == "right.tip_offset.x") {
            new_right_offset.x() = param.as_double(); right_offset_changed = true;
        }
        else if (name == "right.tip_offset.y") {
            new_right_offset.y() = param.as_double(); right_offset_changed = true;
        }
        else if (name == "right.tip_offset.z") {
            new_right_offset.z() = param.as_double(); right_offset_changed = true;
        }
        else {
            RCLCPP_WARN(this->get_logger(), "Parameter '%s' is not dynamically reconfigurable.", name.c_str());
        }
    }

    if (left_offset_changed)  set_left_tip_offset(new_left_offset);
    if (right_offset_changed) set_right_tip_offset(new_right_offset);

    return result;
}

static void update_perf_window(
    std::deque<PerfSample>& window,
    const SolvedPipelineSample& solved,
    std::chrono::steady_clock::time_point now)
{
    window.push_back({
        now,
        solved.keypoint.keypoints.inference_time_ms,
        solved.solution.total_time_ms});
    while (std::chrono::duration<double>(now - window.front().timestamp).count() > PERF_WINDOW_SECONDS)
        window.pop_front();
}

static void log_publish_stats(
    const std::deque<PerfSample>& window,
    const SolvedPipelineSample& solved,
    rclcpp::Logger logger)
{
    double avg_inference_ms = 0.0, avg_tracking_ms = 0.0;
    for (const auto& s : window) {
        avg_inference_ms += s.inference_time_ms;
        avg_tracking_ms  += s.tracking_time_ms;
    }
    avg_inference_ms /= window.size();
    avg_tracking_ms  /= window.size();
    double rate_hz = window.size() / PERF_WINDOW_SECONDS;
    std::string calibration_status = solved.calibration_stopped ? "stopped" : "running";

    RCLCPP_INFO(logger,
        "publish rate:  %.1f Hz | "
        "keypoints: L:%zu, R:%zu, %.1f ms | "
        "tracking: %.1f ms | "
        "calibration: %s",
        rate_hz, 
        solved.keypoint.keypoints.left_keypoints.size(),
        solved.keypoint.keypoints.right_keypoints.size(),
        avg_inference_ms, 
        avg_tracking_ms,
        calibration_status.c_str());
}

void SmootherNode::publish_loop()
{
    auto timeout = std::chrono::milliseconds(1000);

    // Save calibration at the end of a fresh calibration run, skip if we loaded from file
    bool calibration_saved = false;

    std::deque<PerfSample> perf_window;
    auto last_perf_log = std::chrono::steady_clock::now();

    while (rclcpp::ok())
    {
        // Wait for a solved sample to become available from the pipeline
        std::optional<SolvedPipelineSample> solved = pipeline_->wait_for_solved_sample(timeout);

        if (!solved.has_value())
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                5000,
                "Publish loop: no solved sample to publish. Are all topics being published (e.g., camera_info, image_rect, joint_states?)");
            continue;
        }

        publisher_->publish(*solved);

        // Update rolling performance window
        auto now = std::chrono::steady_clock::now();
        update_perf_window(perf_window, *solved, now);

        // Log performance stats once per second
        if (std::chrono::duration<double>(now - last_perf_log).count() >= 1.0) {
            last_perf_log = now;
            log_publish_stats(perf_window, *solved, this->get_logger());
        }

        // If the calibration has been stopped and we haven't saved it yet, save to file
        if (solved->calibration_stopped && !load_calibration_on_start_ && !calibration_saved)
        {
            // Only try once: a failed save must not take down tracking (it used to throw and kill the node)
            calibration_saved = true;
            const std::filesystem::path calibration_file =
                std::filesystem::path(run_dir_) / "calibration" / "smoother" / "smoother_calibration.yaml";
            try {
                std::filesystem::create_directories(calibration_file.parent_path());
                save_calibration_yaml(solved->solution.calibration, calibration_file.string());
                RCLCPP_INFO(this->get_logger(), "Final calibration saved to: %s", calibration_file.c_str());
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(),
                    "Could not save calibration (tracking continues with it locked in): %s. "
                    "Check run_dir (currently '%s').", e.what(), run_dir_.c_str());
            }
        }
    }
}