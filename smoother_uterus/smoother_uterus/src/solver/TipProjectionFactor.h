#pragma once

#include <vector>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

bool project_pose_to_uvz(
    const gtsam::Pose3& tip_pose,
    const gtsam::Cal3_S2& intrinsics,
    gtsam::Point3& uvz,
    gtsam::OptionalJacobian<3, 6> d_uvz_d_tip_pose);

class TipProjectionFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3> {
    using NoiseModelFactorN<gtsam::Pose3>::evaluateError;

public:
    TipProjectionFactor(
        gtsam::Key tip_pose_key,
        const std::vector<gtsam::Vector2>& keypoints,
        gtsam::Cal3_S2 camera_intrinsics,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(const gtsam::Pose3& tip_pose, gtsam::OptionalMatrixType H1) const override;

private:
    std::vector<gtsam::Vector2> keypoints_;
    gtsam::Cal3_S2 camera_intrinsics_;
};
