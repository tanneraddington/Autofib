#include "TipProjectionFactor.h"

#include <gtsam/geometry/PinholeCamera.h>

using namespace gtsam;

namespace {
    constexpr double MIN_DEPTH = 1e-6; // meters; points at or behind this depth are treated as behind the camera
} // namespace

bool project_pose_to_uvz(
    const gtsam::Pose3& tip_pose,
    const Cal3_S2& intrinsics,
    Point3& uvz,
    OptionalJacobian<3, 6> d_uvz_d_tip_pose)
{
    Matrix36 d_p_d_tip_pose;
    Point3 p = tip_pose.translation(d_p_d_tip_pose);

    uvz.z() = p.z();
    bool behind_camera = uvz.z() <= MIN_DEPTH;

    PinholeCamera<Cal3_S2> camera(Pose3::Identity(), intrinsics);

    if (!behind_camera) {
        Matrix23 d_uv_d_p;
        gtsam::Point2 uv = camera.project2(p, std::nullopt, d_uv_d_p);

        uvz.x() = uv.x();
        uvz.y() = uv.y();

        if (d_uvz_d_tip_pose) {
            Matrix3 d_uvz_d_p;
            d_uvz_d_p.topRows<2>() = d_uv_d_p;
            d_uvz_d_p.bottomRows<1>() << 0, 0, 1;

            *d_uvz_d_tip_pose = d_uvz_d_p * d_p_d_tip_pose;
        }
    } else {
        uvz.setZero();
        if (d_uvz_d_tip_pose)
            d_uvz_d_tip_pose->setZero();
    }

    return behind_camera;
}

TipProjectionFactor::TipProjectionFactor(
    Key camera_mount_key,
    Key tip_pose_key,
    const std::vector<gtsam::Vector2>& keypoints,
    Cal3_S2 camera_intrinsics,
    const Pose3& view,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, camera_mount_key, tip_pose_key),
    keypoints_(keypoints),
    camera_intrinsics_(camera_intrinsics),
    view_(view) {}

Vector TipProjectionFactor::evaluateError(
    const Pose3& camera_mount,
    const Pose3& tip_pose,
    OptionalMatrixType H1,
    OptionalMatrixType H2) const
{
    // Endoscope frame -> this image's camera frame: camera_mount * view, then the tip
    Matrix66 d_mount_view_d_mount;
    const Pose3 mount_view = camera_mount.compose(view_, d_mount_view_d_mount);

    Matrix66 d_cam_d_mount_view, d_cam_d_tip;
    const Pose3 tip_in_camera = mount_view.compose(tip_pose, d_cam_d_mount_view, d_cam_d_tip);

    Point3 uvz;
    Matrix36 d_uvz_d_tip_in_camera;
    bool behind_camera = project_pose_to_uvz(tip_in_camera, camera_intrinsics_, uvz, d_uvz_d_tip_in_camera);

    if (keypoints_.empty() || behind_camera) {
        if (H1) *H1 = Matrix26::Zero();
        if (H2) *H2 = Matrix26::Zero();
        return Vector2::Zero();
    }

    // Max mixture approximation: associate to the closest keypoint
    double min_dist_squared = std::numeric_limits<double>::max();
    Vector2 closest_keypoint;
    for (const auto& keypoint : keypoints_) {
        double dist_squared = (uvz.head<2>() - keypoint).squaredNorm();
        if (dist_squared < min_dist_squared) {
            min_dist_squared = dist_squared;
            closest_keypoint = keypoint;
        }
    }

    const Matrix26 d_uv_d_tip_in_camera = d_uvz_d_tip_in_camera.block<2,6>(0, 0);
    if (H1) { *H1 = d_uv_d_tip_in_camera * d_cam_d_mount_view * d_mount_view_d_mount; }
    if (H2) { *H2 = d_uv_d_tip_in_camera * d_cam_d_tip; }
    return uvz.head<2>() - closest_keypoint;
}
