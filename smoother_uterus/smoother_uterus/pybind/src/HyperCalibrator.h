#pragma once

#include "solver/SmootherSolver.h"
#include <memory>


struct HyperCalibratorSolution{
    std::vector<gtsam::Vector6> pose_errors;
    std::vector<gtsam::Vector6> whitened_pose_errors;

    std::vector<gtsam::Vector3> position_errors;
    std::vector<gtsam::Vector3> whitened_position_errors;

    double nll;
};


class HyperCalibrator {
public:
    HyperCalibrator(const std::vector<gtsam::Vector4>& joint_values);

    HyperCalibratorSolution solve(
        const std::optional<std::vector<gtsam::Matrix4>>& tip_pose_meas,
        const std::optional<std::vector<gtsam::Vector3>>& tip_position_meas,
        const std::vector<gtsam::Vector3>& tip_force_meas,
        const gtsam::Vector2& outer_curvature,
        const gtsam::Vector2& inner_curvature,
        const gtsam::Vector6& inner_tube_noise_std,
        const gtsam::Vector6& outer_tube_noise_std,
        double outer_k_bending,
        double inner_k_bending,
        double k_torsion);

private:
    std::unique_ptr<SmootherSolver> solver_;
    std::vector<gtsam::Vector4> joint_values_;
};
