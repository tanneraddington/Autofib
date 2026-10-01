#include "ConstantVelocityFactor.h"

using namespace gtsam;

ConstantVelocityFactor::ConstantVelocityFactor(
    gtsam::Key pose_1_key,
    gtsam::Key pose_2_key,
    gtsam::Key pose_3_key,
    double t1,
    double t2,
    double t3,
    const gtsam::SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_1_key, pose_2_key, pose_3_key),
    t1_(t1),
    t2_(t2),
    t3_(t3)
{}

Vector ConstantVelocityFactor::evaluateError(
    const Pose3& pose_1,
    const Pose3& pose_2,
    const Pose3& pose_3,
    OptionalMatrixType H1,
    OptionalMatrixType H2,
    OptionalMatrixType H3) const
{
    Matrix36 d_p1_d_pose1;
    Vector3 p1 = pose_1.translation(d_p1_d_pose1);

    Matrix36 d_p2_d_pose2;
    Vector3 p2 = pose_2.translation(d_p2_d_pose2);

    Matrix36 d_p3_d_pose3;
    Vector3 p3 = pose_3.translation(d_p3_d_pose3);

    double a = t2_ - t1_;
    double b = t3_ - t2_;
    double c = t3_ - t1_;
    double A = 2.0 / (a * b * c);

    // Finite difference approximation of acceleration
    Vector3 error = (2.0 / (a * b * c)) * (a * p3 - c * p2 + b * p1);

    if (H1) *H1 = A * b * d_p1_d_pose1;
    if (H2) *H2 = A * (-c) * d_p2_d_pose2;
    if (H3) *H3 = A * a * d_p3_d_pose3;

    return error;
}