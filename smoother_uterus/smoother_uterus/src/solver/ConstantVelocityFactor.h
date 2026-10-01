#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>

class ConstantVelocityFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
    using NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>::evaluateError;

public:
    ConstantVelocityFactor(
        gtsam::Key pose_1_key,
        gtsam::Key pose_2_key,
        gtsam::Key pose_3_key,
        double t1,
        double t2,
        double t3,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_1,
        const gtsam::Pose3& pose_2,
        const gtsam::Pose3& pose_3,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3) const override;

private:
    double t1_;
    double t2_;
    double t3_;
};
