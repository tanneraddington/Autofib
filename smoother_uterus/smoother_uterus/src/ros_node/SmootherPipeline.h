#pragma once

#include <atomic>
#include <deque>
#include <optional>
#include <mutex>
#include <thread>
#include <condition_variable>

#include <gtsam/geometry/Cal3_S2.h>
#include <opencv2/opencv.hpp>
#include <gtsam/geometry/Point2.h>
#include <rclcpp/rclcpp.hpp>

#include "solver/SolverTypes.h"
#include "solver/SmootherSolver.h"
#include "KeypointModel.h"

struct RawSample {
    rclcpp::Time stamp;
    cv::Mat image;
    gtsam::Vector4 left_joint_values;
    gtsam::Vector4 right_joint_values;
    Vector3Gaussian left_tip_force_prior;
    Vector3Gaussian right_tip_force_prior;
    bool ignore_keypoints;
};

struct KeypointPipelineSample {
    RawSample raw;
    KeypointInferenceResult keypoints;
};

struct SolvedPipelineSample {
    KeypointPipelineSample keypoint;
    SmootherSolution solution;
    bool calibration_stopped;
};

class SmootherPipeline {
public:
    SmootherPipeline(
        const rclcpp::Logger& logger,
        const gtsam::Cal3_S2& camera_intrinsics,
        const std::string& keypoint_model_path,
        const std::optional<SmootherCalibration>& calibration);

    ~SmootherPipeline();

    // All of these interfaces are called by the ROS node, so they need to be thread safe and fast
    void push_raw_sample(const RawSample& sample);

    std::optional<SolvedPipelineSample> wait_for_solved_sample(std::chrono::milliseconds timeout);

    void stop_calibration();

    void set_tip_offsets(const gtsam::Vector3& left_offset, const gtsam::Vector3& right_offset);

private:
    void keypoint_loop();
    void tracking_loop();
    void calibration_loop();
    void solve_calibration();

    // Blocking reads — return nullopt if stop_threads_ fires
    std::optional<RawSample> wait_for_raw_sample();
    std::optional<KeypointPipelineSample> wait_for_keypoint_sample();

    // Writes to inter-thread mailboxes
    void post_keypoint_sample_to_tracker(const KeypointPipelineSample& sample);
    void post_keypoint_sample_to_calibrator(const KeypointPipelineSample& sample);
    void post_calibration_update_to_tracker(const SmootherCalibration& calibration);
    void post_tracker_solution_to_publisher(SolvedPipelineSample sample);

    // Reads/consumes from inter-thread mailboxes
    void consume_calibration_update(SmootherSolver& solver);
    void consume_tip_offsets(SmootherSolver& solver);

    std::atomic<bool> stop_threads_{false};
    std::atomic<bool> stop_calibration_{false};

    rclcpp::Logger logger_;
    gtsam::Cal3_S2 camera_intrinsics_;
    std::string keypoint_model_path_;
    std::optional<SmootherCalibration> initial_calibration_;

    // ROS node to keypoint thread
    std::mutex raw_mtx_;
    std::condition_variable raw_cv_;
    std::optional<RawSample> raw_mailbox_;

    // Keypoint thread to tracking thread
    std::mutex tracker_mailbox_mtx_;
    std::condition_variable tracker_mailbox_cv_;
    std::optional<KeypointPipelineSample> tracker_mailbox_;

    // Keypoint thread to calibration thread (throttled per-arm queues)
    std::mutex calibrator_dataset_mtx_;
    std::deque<SingleArmSample> calibrator_left_dataset_;
    std::deque<SingleArmSample> calibrator_right_dataset_;
    
    // Calibration thread to tracking thread
    std::mutex latest_calibration_mtx_;
    std::optional<SmootherCalibration> latest_calibration_mailbox_;

    // Tracking thread to publisher (external)
    std::mutex solved_mtx_;
    std::condition_variable solved_cv_;
    std::optional<SolvedPipelineSample> solved_mailbox_;

    // Left/right tip offset mailbox, updated via ROS params
    std::mutex tip_offset_mtx_;
    std::optional<std::array<gtsam::Vector3, 2>> tip_offset_mailbox_; // [left, right]

    std::thread keypoint_thread_;
    std::thread tracker_thread_;
    std::thread calibrator_thread_;
};
