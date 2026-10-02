#pragma once

#include <array>
#include <optional>
#include <vector>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

// Shared structural constant — used in array size declarations across multiple translation units
constexpr size_t NODES_PER_TUBE = 8;

enum class ArmSide { LEFT  = 0, RIGHT = 1 };
enum class ArmTube { OUTER = 0, INNER = 1 };

struct Vector2Gaussian {
    gtsam::Vector2 mean;
    gtsam::Matrix2 cov;
};

struct Vector3Gaussian {
    gtsam::Vector3 mean;
    gtsam::Matrix3 cov;
};

struct Pose3Gaussian {
    gtsam::Matrix4 mean;
    gtsam::Matrix6 cov;
};

struct SingleArmCalibration {
    Pose3Gaussian base_pose;
    Vector2Gaussian outer_curvature;
    Vector2Gaussian inner_curvature;
    size_t num_samples = 0;

    // These fields are for checking if the calibration is valid: between(nominal, solved) < threshold
    gtsam::Vector2 outer_curvature_diff = gtsam::Vector2::Zero();
    gtsam::Vector2 inner_curvature_diff = gtsam::Vector2::Zero();
    gtsam::Vector6 base_pose_diff       = gtsam::Vector6::Zero();
    bool is_valid = false;
};

// Everything the solver estimates is in the endoscope tip frame (TF: smoother_uterus/endo), which is fixed to
// the scope and doesn't move when the view angle changes. The camera's pose in it depends on the view angle:
//   camera_from_endo(angle) = camera_mount * view(angle)
// (see SmootherSolver.cpp). Base poses here are therefore valid at every view angle.
struct SmootherCalibration {
    SingleArmCalibration left_arm;
    SingleArmCalibration right_arm;

    // Lens/pivot frame in the camera optical frame. Absent -> the solver holds it at its nominal value.
    std::optional<Pose3Gaussian> camera_mount;
    gtsam::Vector6 camera_mount_diff = gtsam::Vector6::Zero();  // Logmap(nominal.between(solved)), for validity checks

    double total_time_ms = 0.0;
    bool is_valid = false;
};

struct SingleArmSample {
    SingleArmSample() = default;
    SingleArmSample(
        double time_seconds,
        const gtsam::Vector4& joint_values,
        const std::vector<gtsam::Vector2>& keypoints,
        const Vector3Gaussian& tip_force,
        const std::optional<Vector3Gaussian>& tip_position_meas,
        std::optional<double> camera_angle = std::nullopt);

    double time_seconds;
    gtsam::Vector4 joint_values;
    std::vector<gtsam::Vector2> keypoints;
    Vector3Gaussian tip_force;
    std::optional<Vector3Gaussian> tip_position_meas;  // in the endoscope tip frame
    std::optional<double> camera_angle;                // view angle (rad) for this image; nullopt = nominal CAMERA_ANGLE
};

// Poses, force and jac_tip_pose are in the endoscope tip frame; the *_uvz projections use this sample's camera.
struct SingleArmMarginals {
    std::array<Pose3Gaussian, NODES_PER_TUBE> outer_tube_poses;
    std::array<Pose3Gaussian, NODES_PER_TUBE> inner_tube_poses;

    Pose3Gaussian tip_pose;
    Vector3Gaussian tip_force;
    gtsam::Matrix64 jac_tip_pose;

    std::array<Vector3Gaussian, NODES_PER_TUBE> outer_tube_uvz;
    std::array<Vector3Gaussian, NODES_PER_TUBE> inner_tube_uvz;
    Vector3Gaussian tip_uvz;

    // Maps endoscope-frame poses into this sample's camera frame: pose_in_camera = camera_from_endo * pose
    gtsam::Matrix4 camera_from_endo = gtsam::Matrix4::Identity();
};

struct SmootherSolution {
    std::vector<SingleArmMarginals> left_arm;
    std::vector<SingleArmMarginals> right_arm;
    SmootherCalibration calibration;

    double error;
    int iterations;
    double build_time_ms;
    double optimize_time_ms;
    double extract_time_ms;
    double total_time_ms;
};
