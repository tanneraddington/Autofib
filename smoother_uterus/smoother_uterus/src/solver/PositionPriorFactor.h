#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>

class PositionPriorFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3> {
    using NoiseModelFactorN<gtsam::Pose3>::evaluateError;

public:
    PositionPriorFactor(gtsam::Key pose_key, gtsam::Vector3 position_meas, const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, gtsam::OptionalMatrixType H1) const override;

private:
    gtsam::Vector3 position_meas_;
};



