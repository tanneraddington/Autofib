#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>


using CosseratTwistBase = gtsam::NoiseModelFactorN<
    gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector2>;


class CosseratTwistFactor: public CosseratTwistBase {
    using CosseratTwistBase::evaluateError;

public:
    CosseratTwistFactor(
        gtsam::Key pose_0_key,
        gtsam::Key pose_1_key,
        gtsam::Key pose_tip_key,
        gtsam::Key tip_force_key,
        gtsam::Key curvature_key,
        double ds,
        const gtsam::Matrix6& K_inv,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_0,
        const gtsam::Pose3& pose_1,
        const gtsam::Pose3& pose_tip,
        const gtsam::Vector3& tip_force,
        const gtsam::Vector2& curvature,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3,
        gtsam::OptionalMatrixType H4,
        gtsam::OptionalMatrixType H5) const override;

private:
    gtsam::Vector6 propagate_wrench_backward(
        const gtsam::Pose3& pose,
        const gtsam::Pose3& tip_pose,
        const gtsam::Vector6& tip_wrench,
        gtsam::OptionalMatrixType d_wrench_d_pose,
        gtsam::OptionalMatrixType d_wrench_d_tip_pose,
        gtsam::OptionalMatrixType d_wrench_d_tip_wrench) const;

    double ds_;  // segment length
    gtsam::Matrix66 K_inv_;  // Assuming constant stiffness inverse per factor
};
