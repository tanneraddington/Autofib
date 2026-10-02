#pragma once

#include <vector>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

// Projects a pose given in the CAMERA frame to pixels (u, v) and depth z. Returns true if behind the camera.
bool project_pose_to_uvz(
    const gtsam::Pose3& tip_pose,
    const gtsam::Cal3_S2& intrinsics,
    gtsam::Point3& uvz,
    gtsam::OptionalJacobian<3, 6> d_uvz_d_tip_pose);

// Keypoint factor on the tube end, which is estimated in the endoscope tip frame. For each image:
//   tip_in_camera = camera_mount * view * tip_pose
// camera_mount is a variable (calibrated, shared by all images); view is that image's known view-angle
// rotation, a constant here. Associates to the closest detected keypoint (max-mixture approximation).
class TipProjectionFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
    using NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3>::evaluateError;

public:
    TipProjectionFactor(
        gtsam::Key camera_mount_key,
        gtsam::Key tip_pose_key,
        const std::vector<gtsam::Vector2>& keypoints,
        gtsam::Cal3_S2 camera_intrinsics,
        const gtsam::Pose3& view,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& camera_mount,
        const gtsam::Pose3& tip_pose,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2) const override;

private:
    std::vector<gtsam::Vector2> keypoints_;
    gtsam::Cal3_S2 camera_intrinsics_;
    gtsam::Pose3 view_;
};
