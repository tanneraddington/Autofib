#include "CosseratTwistFactor.h"

using namespace gtsam;

CosseratTwistFactor::CosseratTwistFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key pose_tip_key,
    Key tip_force_key,
    Key curvature_key,
    double ds,
    const Matrix66& K_inv,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_0_key, pose_1_key, pose_tip_key, tip_force_key, curvature_key),
    ds_(ds),
    K_inv_(K_inv) {}

Vector CosseratTwistFactor::evaluateError(
    const Pose3& pose_0,
    const Pose3& pose_1,
    const Pose3& pose_tip,
    const Vector3& tip_force,
    const Vector2& curvature,
    OptionalMatrixType H1,
    OptionalMatrixType H2,
    OptionalMatrixType H3,
    OptionalMatrixType H4,
    OptionalMatrixType H5) const
{
    Matrix6 d_delta_d_pose_0, d_delta_d_pose_1;
    Pose3 delta = pose_0.between(pose_1, d_delta_d_pose_0, d_delta_d_pose_1);

    Matrix6 d_twist_d_delta;
    Vector6 twist = Pose3::Logmap(delta, d_twist_d_delta);

    Matrix36 d_R_tip_d_pose_tip;
    Rot3 R_tip = pose_tip.rotation(d_R_tip_d_pose_tip);

    Matrix3 d_force_body_d_R_tip, d_force_body_d_force;
    Vector3 tip_force_body = R_tip.unrotate(tip_force, d_force_body_d_R_tip, d_force_body_d_force);

    Vector6 tip_wrench = Vector6::Zero();
    tip_wrench.tail<3>() = tip_force_body;

    Matrix63 d_tip_wrench_d_force_body = Matrix63::Zero();
    d_tip_wrench_d_force_body.bottomRows<3>() = Matrix3::Identity();

    // d_stress_N_d_tip_wrench feeds both H3 (via force body chain) and H4, so compute it when either is needed.
    const bool need_tip_wrench_jac = H3 || H4;

    Matrix d_stress_0_d_pose_0, d_stress_0_d_pose_tip, d_stress_0_d_tip_wrench;
    Vector6 stress_0 = propagate_wrench_backward(
        pose_0,
        pose_tip,
        tip_wrench,
        H1 ? &d_stress_0_d_pose_0 : nullptr,
        H3 ? &d_stress_0_d_pose_tip : nullptr,
        need_tip_wrench_jac ? &d_stress_0_d_tip_wrench : nullptr);

    Matrix d_stress_1_d_pose_1, d_stress_1_d_pose_tip, d_stress_1_d_tip_wrench;
    Vector6 stress_1 = propagate_wrench_backward(
        pose_1,
        pose_tip,
        tip_wrench,
        H2 ? &d_stress_1_d_pose_1 : nullptr,
        H3 ? &d_stress_1_d_pose_tip : nullptr,
        need_tip_wrench_jac ? &d_stress_1_d_tip_wrench : nullptr);

    Vector6 stress = 0.5 * (stress_0 + stress_1);  // Midpoint rule

    Vector6 nominal_strain;
    nominal_strain.head<3>() << curvature.x(), curvature.y(), 0.0;  // First three elements from curvature
    nominal_strain.tail<3>() << 0.0, 0.0, 1.0;  // Last three elements: p_dot = e_3

    Vector6 predicted_twist = ds_ * (K_inv_ * stress + nominal_strain);
    Vector6 twist_error = predicted_twist - twist;

    if (H1) {
        *H1 = ds_ * K_inv_ * 0.5 * d_stress_0_d_pose_0 - d_twist_d_delta * d_delta_d_pose_0;
    }

    if (H2) {
        *H2 = ds_ * K_inv_ * 0.5 * d_stress_1_d_pose_1 - d_twist_d_delta * d_delta_d_pose_1;
    }

    if (H3) {
        *H3 = ds_ * K_inv_ * 0.5 * (
            d_stress_0_d_pose_tip + d_stress_0_d_tip_wrench * d_tip_wrench_d_force_body * d_force_body_d_R_tip * d_R_tip_d_pose_tip +
            d_stress_1_d_pose_tip + d_stress_1_d_tip_wrench * d_tip_wrench_d_force_body * d_force_body_d_R_tip * d_R_tip_d_pose_tip);
    }

    if (H4) {
        *H4 = ds_ * K_inv_ * 0.5 * (
            d_stress_0_d_tip_wrench * d_tip_wrench_d_force_body * d_force_body_d_force +
            d_stress_1_d_tip_wrench * d_tip_wrench_d_force_body * d_force_body_d_force);
    }

    if (H5) {
        *H5 = Matrix62::Zero();
        H5->block<2,2>(0,0) = ds_ * Matrix2::Identity();  // del_twist_del_curvature
    }

    return twist_error;
}

Vector6 CosseratTwistFactor::propagate_wrench_backward(
    const Pose3& pose,
    const Pose3& tip_pose,
    const Vector6& tip_wrench,
    OptionalMatrixType d_wrench_d_pose,
    OptionalMatrixType d_wrench_d_tip_pose,
    OptionalMatrixType d_wrench_d_tip_wrench) const
{
    Matrix66 d_tip_pose_inv_d_tip_pose;
    Matrix66 d_delta_d_tip_pose_inv, d_delta_d_pose;
    Matrix66 d_wrench_d_delta, d_wrench_d_input_wrench;

    Pose3 tip_pose_inv = tip_pose.inverse(
        (d_wrench_d_tip_pose ? &d_tip_pose_inv_d_tip_pose : nullptr));

    Pose3 delta = tip_pose_inv.compose(pose,
        (d_wrench_d_tip_pose ? &d_delta_d_tip_pose_inv : nullptr),
        (d_wrench_d_pose ? &d_delta_d_pose : nullptr));

    Vector6 wrench = delta.AdjointTranspose(
        tip_wrench,
        (d_wrench_d_pose || d_wrench_d_tip_pose ? &d_wrench_d_delta : nullptr),
        (d_wrench_d_tip_wrench ? &d_wrench_d_input_wrench : nullptr));

    if (d_wrench_d_pose) {
        *d_wrench_d_pose = d_wrench_d_delta * d_delta_d_pose;
    }
    if (d_wrench_d_tip_pose) {
        *d_wrench_d_tip_pose = d_wrench_d_delta * d_delta_d_tip_pose_inv * d_tip_pose_inv_d_tip_pose;
    }
    if (d_wrench_d_tip_wrench) {
        *d_wrench_d_tip_wrench = d_wrench_d_input_wrench;
    }

    return wrench;
}
