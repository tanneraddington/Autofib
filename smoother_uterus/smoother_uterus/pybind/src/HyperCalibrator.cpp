#include "HyperCalibrator.h"

#include <deque>
#include <memory>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>

using namespace gtsam;


HyperCalibrator::HyperCalibrator(const std::vector<Vector4>& joint_values)
    : solver_(std::make_unique<SmootherSolver>(Cal3_S2(800.0, 800.0, 0.0, 320.0, 240.0))),
      joint_values_(joint_values)
{}

HyperCalibratorSolution HyperCalibrator::solve(
    const std::optional<std::vector<Matrix4>>& tip_pose_meas,
    const std::optional<std::vector<Vector3>>& tip_position_meas,
    const std::vector<Vector3>& tip_force_meas,
    const Vector2& outer_curvature,
    const Vector2& inner_curvature,
    const Vector6& inner_tube_noise_std,
    const Vector6& outer_tube_noise_std,
    double outer_k_bending,
    double inner_k_bending,
    double k_torsion)
{
    if (tip_pose_meas && tip_position_meas) {
        throw std::invalid_argument("You cant use both pose and position measurements!");
    }

    std::deque<SingleArmSample> left_samples;
    for (size_t t = 0; t < joint_values_.size(); t++) {
        SingleArmSample sample;
        sample.time_seconds = static_cast<double>(t);
        sample.joint_values = joint_values_[t];
        sample.tip_force.mean = tip_force_meas[t];
        sample.tip_force.cov = 0.01 * 0.01 * Matrix3::Identity();
        left_samples.push_back(sample);
    }

    SmootherCalibration c;
    c.left_arm.base_pose.mean = Matrix4::Identity();
    c.left_arm.base_pose.cov = 1e-8 * Matrix6::Identity();
    c.left_arm.outer_curvature.mean = outer_curvature;
    c.left_arm.outer_curvature.cov = 1e-4 * Matrix2::Identity();
    c.left_arm.inner_curvature.mean = inner_curvature;
    c.left_arm.inner_curvature.cov = 1e-4 * Matrix2::Identity();
    c.right_arm = c.left_arm;

    solver_->set_calibration(c);
    solver_->set_inner_tube_noise_std(inner_tube_noise_std);
    solver_->set_outer_tube_noise_std(outer_tube_noise_std);
    solver_->set_stiffness_params(outer_k_bending, inner_k_bending, k_torsion);

    std::deque<SingleArmSample> right_samples;
    SmootherSolution solved = solver_->solve(left_samples, right_samples);

    HyperCalibratorSolution solution;
    solution.nll = 0.0;

    if (tip_pose_meas) {
        for (size_t t = 0; t < tip_pose_meas.value().size(); t++) {
            Pose3 pose_mean = Pose3(solved.left_arm[t].tip_pose.mean);
            Matrix6 pose_cov = solved.left_arm[t].tip_pose.cov;
            Pose3 pose_meas = Pose3(tip_pose_meas.value()[t]);

            Matrix6 d_delta_d_tip_pose;
            Pose3 delta = pose_mean.between(pose_meas, d_delta_d_tip_pose);
            Matrix6 d_error_d_delta;
            Vector6 error = Pose3::Logmap(delta, d_error_d_delta);
            Matrix6 d_error_d_tip_pose = d_error_d_delta * d_delta_d_tip_pose;
            Matrix6 error_cov = d_error_d_tip_pose * pose_cov * d_error_d_tip_pose.transpose();

            Eigen::LLT<Matrix6> llt(error_cov);
            Matrix6 L = llt.matrixL();
            Vector6 white = L.triangularView<Eigen::Lower>().solve(error);

            solution.nll += 0.5 * white.squaredNorm() + L.diagonal().array().log().sum();
            solution.pose_errors.push_back(error);
            solution.whitened_pose_errors.push_back(white);
        }
    }

    if (tip_position_meas) {
        for (size_t t = 0; t < tip_position_meas.value().size(); t++) {
            Pose3 pose = Pose3(solved.left_arm[t].tip_pose.mean);
            Vector3 p_mean = pose.translation();
            Matrix3 R = pose.rotation().matrix();
            Matrix3 p_cov_local = solved.left_arm[t].tip_pose.cov.block<3,3>(3,3);
            Matrix3 p_cov = R * p_cov_local * R.transpose();
            Vector3 p_meas = tip_position_meas.value()[t];

            Vector3 error = p_mean - p_meas;
            Matrix3 S = p_cov + 0.0001 * 0.0001 * Matrix3::Identity();

            Eigen::LLT<Matrix3> llt(S);
            Matrix3 L = llt.matrixL();
            Vector3 white = L.triangularView<Eigen::Lower>().solve(error);

            solution.nll += 0.5 * white.squaredNorm() + L.diagonal().array().log().sum();
            solution.position_errors.push_back(error);
            solution.whitened_position_errors.push_back(white);
        }
    }

    return solution;
}
