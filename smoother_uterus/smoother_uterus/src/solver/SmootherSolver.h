#pragma once

#include <deque>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>

#include "SolverTypes.h"

class SmootherSolver {
public:
    SmootherSolver(const gtsam::Cal3_S2& camera_intrinsics);

    SmootherSolution solve(
        const std::deque<SingleArmSample>& left_samples,
        const std::deque<SingleArmSample>& right_samples,
        int max_iterations = 100,
        bool two_stage = false);

    void set_tip_accel_prior_std(double accel_prior_std);
    void set_calibration(const SmootherCalibration& calibration);
    void set_inner_tube_noise_std(const gtsam::Vector6& twist_std);
    void set_outer_tube_noise_std(const gtsam::Vector6& twist_std);
    void set_stiffness_params(double outer_k_bending, double inner_k_bending, double k_torsion);
    void set_left_tip_offset(const gtsam::Vector3& offset);
    void set_right_tip_offset(const gtsam::Vector3& offset);
    void set_pixel_meas_std(double pixel_meas_std);

    ~SmootherSolver() = default;

private:
    void init_noise_models();
    void init_values();
    void build_graph(bool include_keypoints = true);
    void add_calibration_factors();
    void add_single_sample_factors(ArmSide side, int time_idx, bool include_keypoints = true);
    void add_geometry_prior_factors();
    void add_motion_prior_factors();
    void extract_calibration(SmootherSolution& solution);
    void extract_single_arm(ArmSide side, int time_idx, SingleArmMarginals& solution);
    void extract_solution(SmootherSolution& solution);

    const std::deque<SingleArmSample>& samples_for(ArmSide side) const;

    // Transient — set at the start of solve(), cleared before returning.
    const std::deque<SingleArmSample>* left_samples_ = nullptr;
    const std::deque<SingleArmSample>* right_samples_ = nullptr;

    gtsam::Vector3 left_tip_offset_ = gtsam::Vector3::Zero();
    gtsam::Vector3 right_tip_offset_ = gtsam::Vector3::Zero();

    gtsam::Cal3_S2 camera_intrinsics_;

    std::optional<SmootherCalibration> calibration_ = std::nullopt;

    // Noise models — built once in build_noise_models(), called from constructor.
    // Setters that affect noise params rebuild their model(s) directly.
    gtsam::SharedNoiseModel outer_tube_noise_model_;          // set_outer_tube_noise_std
    gtsam::SharedNoiseModel inner_tube_noise_model_;          // set_inner_tube_noise_std
    gtsam::SharedNoiseModel pixel_noise_model_;               // set_pixel_meas_std
    gtsam::SharedNoiseModel small_curvature_noise_model_;     // no setter (SMALL_CURVATURE_STD is a constant)
    gtsam::SharedNoiseModel tip_accel_prior_noise_model_;     // set_tip_accel_prior_std (null until called)
    gtsam::SharedNoiseModel lens_pose_noise_model_;           // no setter
    gtsam::SharedNoiseModel lens_to_endo_tip_noise_model_;    // no setter
    gtsam::SharedNoiseModel endo_tip_to_endo_base_noise_model_; // no setter
    gtsam::SharedNoiseModel arm_base_to_endo_base_noise_model_; // no setter
    gtsam::SharedNoiseModel outer_curvature_prior_noise_model_; // no setter
    gtsam::SharedNoiseModel inner_curvature_prior_noise_model_; // no setter
    gtsam::SharedNoiseModel small_pose_noise_model_;        // no setter

    // GTSAM factor graph state
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    // Stiffness parameters
    gtsam::Matrix6 outer_K_inv_;
    gtsam::Matrix6 inner_K_inv_;
    gtsam::Matrix6 torsion_K_inv_;
};
