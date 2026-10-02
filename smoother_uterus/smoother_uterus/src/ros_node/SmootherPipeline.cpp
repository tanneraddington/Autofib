#include "SmootherPipeline.h"
#include "solver/SmootherSolver.h"

#include <cmath>
#include <limits>
#include <optional>
#include <chrono>

namespace {
    constexpr size_t TRACKER_LAG_WINDOW_SIZE = 11;
    constexpr double TRACKER_DT_SECONDS = 0.1;
    constexpr size_t CALIBRATOR_LAG_WINDOW_SIZE = std::numeric_limits<size_t>::max();
    constexpr double CALIBRATOR_DT_SECONDS = 0.5;
    constexpr double HOME_INNER_EXTENSION_THRESH = 0.005;  // Only add samples to calibration when arm is far out from home
    constexpr double TIP_ACCEL_PRIOR_STD = 0.02;  // Tuned to allow reasonable acceleration but reject huge keypoint jumps
    constexpr size_t MIN_SAMPLES_FOR_CALIBRATION = 100;  // Tuned to avoid bad calibration solves with too few samples
    constexpr int MAX_SOLVER_ITERATIONS = 25;  // Tuned to avoid long solves with diminishing returns
    constexpr std::chrono::milliseconds CALIBRATION_SLEEP_DURATION{1000};
}

static void push_sample_with_keypoints_throttle(
    std::deque<SingleArmSample>& history,
    const SingleArmSample& sample,
    double dt_seconds,
    size_t max_size)
{
    // Only add to history if we have keypoints and its been long enough since the last sample (or its empty)
    bool has_keypoints = !sample.keypoints.empty();
    bool long_enough_since_last = history.empty() || sample.time_seconds - history.back().time_seconds >= dt_seconds;

    if (has_keypoints && long_enough_since_last)
        history.push_back(sample);

    while (history.size() > max_size) history.pop_front();
}

static SingleArmSample make_left_arm_sample(const KeypointPipelineSample& s)
{
    return {s.raw.stamp.seconds(), s.raw.left_joint_values,
            s.keypoints.left_keypoints, s.raw.left_tip_force_prior, std::nullopt, s.raw.camera_angle};
}

static SingleArmSample make_right_arm_sample(const KeypointPipelineSample& s)
{
    return {s.raw.stamp.seconds(), s.raw.right_joint_values,
            s.keypoints.right_keypoints, s.raw.right_tip_force_prior, std::nullopt, s.raw.camera_angle};
}

static std::optional<SmootherSolution> try_solver_solve(
    SmootherSolver& solver,
    const rclcpp::Logger& logger,
    const char* loop_name,
    const std::deque<SingleArmSample>& left_samples,
    const std::deque<SingleArmSample>& right_samples,
    bool two_stage = false)
{
    try {
        return solver.solve(left_samples, right_samples, MAX_SOLVER_ITERATIONS, two_stage);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "%s solve failed: %s", loop_name, e.what());
    } catch (...) {
        RCLCPP_ERROR(logger, "%s solve failed: %s", loop_name, "unknown exception");
    }

    return std::nullopt;
}

SmootherPipeline::SmootherPipeline(
    const rclcpp::Logger& logger,
    const gtsam::Cal3_S2& camera_intrinsics,
    const std::string& keypoint_model_path,
    const std::optional<SmootherCalibration>& calibration)
:
    logger_(logger),
    camera_intrinsics_(camera_intrinsics),
    keypoint_model_path_(keypoint_model_path),
    initial_calibration_(calibration)
{
    // If a calibration is already loaded, disable the calibration loop before threads start
    if (initial_calibration_)
        stop_calibration_ = true;

    keypoint_thread_   = std::thread(&SmootherPipeline::keypoint_loop,     this);
    tracker_thread_    = std::thread(&SmootherPipeline::tracking_loop,     this);
    calibrator_thread_ = std::thread(&SmootherPipeline::calibration_loop,  this);
}

SmootherPipeline::~SmootherPipeline()
{
    stop_threads_ = true;

    raw_cv_.notify_all();
    tracker_mailbox_cv_.notify_all();
    solved_cv_.notify_all();

    if (keypoint_thread_.joinable())   keypoint_thread_.join();
    if (tracker_thread_.joinable())    tracker_thread_.join();
    if (calibrator_thread_.joinable()) calibrator_thread_.join();
}

void SmootherPipeline::stop_calibration()
{
    // The calibrator runs one final solve with every sample collected, posts it, then locks in. Locking in
    // immediately would keep whatever solve last finished, which can miss the end of a long calibration run.
    stop_requested_ = true;
}

void SmootherPipeline::push_raw_sample(const RawSample& sample)
{
    {
        std::lock_guard lock(raw_mtx_);
        raw_mailbox_ = sample;
    }
    raw_cv_.notify_one();
}

void SmootherPipeline::set_tip_offsets(const gtsam::Vector3& left_offset, const gtsam::Vector3& right_offset)
{
    std::lock_guard lock(tip_offset_mtx_);
    tip_offset_mailbox_ = {left_offset, right_offset};
}

std::optional<SolvedPipelineSample> SmootherPipeline::wait_for_solved_sample(
    std::chrono::milliseconds timeout)
{
    std::unique_lock lock(solved_mtx_);
    bool notified = solved_cv_.wait_for(lock, timeout,
        [this] { return stop_threads_ || solved_mailbox_.has_value(); });
    if (!notified || stop_threads_) return std::nullopt;
    return std::exchange(solved_mailbox_, std::nullopt);
}

std::optional<RawSample> SmootherPipeline::wait_for_raw_sample()
{
    std::unique_lock lock(raw_mtx_);
    raw_cv_.wait(lock, [this] { return stop_threads_ || raw_mailbox_.has_value(); });
    if (stop_threads_) return std::nullopt;
    return std::exchange(raw_mailbox_, std::nullopt);
}

std::optional<KeypointPipelineSample> SmootherPipeline::wait_for_keypoint_sample()
{
    std::unique_lock lock(tracker_mailbox_mtx_);
    tracker_mailbox_cv_.wait(lock, [this] { return stop_threads_ || tracker_mailbox_.has_value(); });
    if (stop_threads_) return std::nullopt;
    return std::exchange(tracker_mailbox_, std::nullopt);
}

void SmootherPipeline::post_keypoint_sample_to_tracker(const KeypointPipelineSample& sample)
{
    {
        std::lock_guard lock(tracker_mailbox_mtx_);
        tracker_mailbox_ = sample;
    }
    tracker_mailbox_cv_.notify_one();
}

void SmootherPipeline::post_keypoint_sample_to_calibrator(const KeypointPipelineSample& sample)
{
    std::lock_guard lock(calibrator_dataset_mtx_);
    push_sample_with_keypoints_throttle(calibrator_left_dataset_,  make_left_arm_sample(sample),  CALIBRATOR_DT_SECONDS, CALIBRATOR_LAG_WINDOW_SIZE);
    push_sample_with_keypoints_throttle(calibrator_right_dataset_, make_right_arm_sample(sample), CALIBRATOR_DT_SECONDS, CALIBRATOR_LAG_WINDOW_SIZE);
}

void SmootherPipeline::post_calibration_update_to_tracker(const SmootherCalibration& calibration)
{
    std::lock_guard lock(latest_calibration_mtx_);
    latest_calibration_mailbox_ = calibration;
}

void SmootherPipeline::consume_calibration_update(SmootherSolver& solver)
{
    std::lock_guard lock(latest_calibration_mtx_);
    if (latest_calibration_mailbox_) {
        solver.set_calibration(*latest_calibration_mailbox_);
        latest_calibration_mailbox_.reset();
    }
}

void SmootherPipeline::consume_tip_offsets(SmootherSolver& solver)
{
    std::lock_guard lock(tip_offset_mtx_);
    if (tip_offset_mailbox_) {
        solver.set_left_tip_offset((*tip_offset_mailbox_)[0]);
        solver.set_right_tip_offset((*tip_offset_mailbox_)[1]);
        tip_offset_mailbox_.reset();
    }
}

void SmootherPipeline::post_tracker_solution_to_publisher(SolvedPipelineSample sample)
{
    {
        std::lock_guard lock(solved_mtx_);
        solved_mailbox_ = std::move(sample);
    }
    solved_cv_.notify_one();
}

void SmootherPipeline::keypoint_loop()
{
    KeypointModel keypoint_model(logger_, keypoint_model_path_);

    while (!stop_threads_) {
        auto raw_sample = wait_for_raw_sample();
        if (!raw_sample) break;

        KeypointInferenceResult keypoints;
        if (!raw_sample->ignore_keypoints) {
            // An exception here would otherwise terminate the whole node from this thread
            try {
                keypoints = keypoint_model.process_image(raw_sample->image);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(logger_, "Keypoint inference failed, skipping keypoints for this frame: %s", e.what());
            }
        }

        // Remove keypoints from arms that are retracted to avoid bad keypoint detections when the arm is home
        if (raw_sample->left_joint_values[2] < HOME_INNER_EXTENSION_THRESH)
            keypoints.left_keypoints.clear();
        if (raw_sample->right_joint_values[2] < HOME_INNER_EXTENSION_THRESH)
            keypoints.right_keypoints.clear();
        
        KeypointPipelineSample keypoint_sample{*raw_sample, keypoints};
        post_keypoint_sample_to_tracker(keypoint_sample);
        post_keypoint_sample_to_calibrator(keypoint_sample);
    }
}

void SmootherPipeline::tracking_loop()
{
    SmootherSolver tracker(camera_intrinsics_);
    tracker.set_tip_accel_prior_std(TIP_ACCEL_PRIOR_STD);

    if (initial_calibration_)
        tracker.set_calibration(*initial_calibration_);

    std::deque<SingleArmSample> left_history, right_history;

    // This loop has to be fast for real time publishing of the current state estimation
    while (!stop_threads_) {
        auto original_keypoint_sample = wait_for_keypoint_sample();
        if (!original_keypoint_sample) break;
        
        // Copy original sample, since we want to use original for publishing but may modify it for tracker solving
        auto keypoint_sample = *original_keypoint_sample;

        // Read once: if set, the final calibration was already posted, so the consume below picks it up and
        // the solution published as "locked in" (and saved) uses it.
        const bool calibration_stopped = stop_calibration_;

        // If we are still calibrating, disregard keypoints to avoid ill-posed solutions with bad initial calibration
        if (!calibration_stopped) {
            keypoint_sample.keypoints.left_keypoints.clear();
            keypoint_sample.keypoints.right_keypoints.clear();
        }

        auto left_arm_sample  = make_left_arm_sample(keypoint_sample);
        auto right_arm_sample = make_right_arm_sample(keypoint_sample);

        // Update tracker with latest parameters if available
        consume_calibration_update(tracker);
        consume_tip_offsets(tracker);

        // Build solve inputs: history with keypoints in front, current time sample (regardless of keypoints) at back
        auto left_solve_samples  = left_history;  left_solve_samples.push_back(left_arm_sample);
        auto right_solve_samples = right_history; right_solve_samples.push_back(right_arm_sample);

        // Go ahead and update the history with current sample respecting keypoints and throttle
        push_sample_with_keypoints_throttle(left_history,  left_arm_sample,  TRACKER_DT_SECONDS, TRACKER_LAG_WINDOW_SIZE - 1);
        push_sample_with_keypoints_throttle(right_history, right_arm_sample, TRACKER_DT_SECONDS, TRACKER_LAG_WINDOW_SIZE - 1);

        // Try to solve the tracker, will catch and log any exceptions and return nullopt if it fails
        std::optional<SmootherSolution> solution = try_solver_solve(tracker, logger_, "TRACKER", left_solve_samples, right_solve_samples);

        // Only publish if we got a solution, otherwise just wait for the next sample and try again
        // Use origianl keypoint sample for publishing so we can see the keypoints even if not actually used 
        if (solution)
            post_tracker_solution_to_publisher({*original_keypoint_sample, *solution, calibration_stopped});
    }
}

void SmootherPipeline::solve_calibration()
{
    // New calibrator on each solve to NOT use warm start for calibration, since we want to get out of local minima
    SmootherSolver calibrator(camera_intrinsics_);

    // Stop if requested, never solve calibration if stop calibration 
    if (stop_threads_ || stop_calibration_)
        return;

    // Copy the current calibrator dataset under lock
    std::deque<SingleArmSample> left_samples, right_samples;
    {
        std::lock_guard lock(calibrator_dataset_mtx_);
        left_samples  = calibrator_left_dataset_;
        right_samples = calibrator_right_dataset_;
    }

    // Don't solve unless we have enough samples, avoids ill posed exceptions
    if (left_samples.size() + right_samples.size() < MIN_SAMPLES_FOR_CALIBRATION)
        return;

    // Try to solve the calibration, will catch and log any exceptions and return nullopt if it fails
    std::optional<SmootherSolution> solution = try_solver_solve(
        calibrator, logger_, "CALIBRATOR", left_samples, right_samples, /*two_stage=*/true);
    
    if (!solution)
        return;

    const auto& lh = solution->calibration.left_arm;
    const auto& rh = solution->calibration.right_arm;
    const auto& cam = solution->calibration.camera_mount_diff;

    RCLCPP_INFO(logger_,
        "CALIBRATOR: new calibration computed, see normed differences from nominal below\n"
        "  left:  valid: %s, outer_curv=%.1f  inner_curv=%.2f  rot=%.1fdeg\n"
        "  right: valid: %s, outer_curv=%.1f  inner_curv=%.2f  rot=%.1fdeg\n"
        "  camera mount: tilt=%.1fdeg  roll=%.1fdeg  pivot offset=%.1fmm",
        lh.is_valid ? "OK  " : "FAIL",
        lh.outer_curvature_diff.norm(), lh.inner_curvature_diff.norm(),
        lh.base_pose_diff.head<3>().norm() * 180.0 / M_PI,
        rh.is_valid ? "OK  " : "FAIL",
        rh.outer_curvature_diff.norm(), rh.inner_curvature_diff.norm(),
        rh.base_pose_diff.head<3>().norm() * 180.0 / M_PI,
        cam.head<2>().norm() * 180.0 / M_PI, std::abs(cam[2]) * 180.0 / M_PI,
        cam.tail<3>().norm() * 1000.0);

    if (!solution->calibration.is_valid) {
        RCLCPP_WARN(logger_, "CALIBRATOR calibration rejected, solution too far from nominal, see above");
        return;
    }

    post_calibration_update_to_tracker(solution->calibration);
}

void SmootherPipeline::calibration_loop()
{
    // Just keep trying to solve the calibration as new samples arrive
    // This loop doesnt have to be fast or interruptible since it runs in the background
    while (!stop_threads_ && !stop_calibration_) {
        std::this_thread::sleep_for(CALIBRATION_SLEEP_DURATION);

        if (stop_requested_) {
            size_t num_samples;
            {
                std::lock_guard lock(calibrator_dataset_mtx_);
                num_samples = calibrator_left_dataset_.size() + calibrator_right_dataset_.size();
            }
            RCLCPP_INFO(logger_, "CALIBRATOR: stop requested, final solve with all %zu samples before locking in...", num_samples);
            solve_calibration();
            stop_calibration_ = true;  // after the final calibration is posted to the tracker
            RCLCPP_INFO(logger_, "CALIBRATOR: calibration locked in.");
            break;
        }

        solve_calibration();
    }
}
