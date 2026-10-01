#include "SmootherSolver.h"

#include <cmath>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include "gtsam/geometry/Point3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/DoglegOptimizer.h"

#include "TipProjectionFactor.h"
#include "CosseratTwistFactor.h"
#include "PositionPriorFactor.h"
#include "ConstantVelocityFactor.h"

using namespace gtsam;

// All solver-internal physical constants and key helpers live here — not in SolverTypes.h which is for shared types only.
namespace {
    // Numerical noise values used to constrain DOF but not overly so as to cause numerical issues
    constexpr double SMALL_ROTATION_STD  = 1e-3;
    constexpr double SMALL_POSITION_STD  = 1e-5;
    constexpr double SMALL_CURVATURE_STD = 1e-2;

    // Tube noise model defaults (override via set_inner/outer_tube_noise_std)
    constexpr double INNER_TUBE_ROT_XY_STD = 0.044;
    constexpr double INNER_TUBE_ROT_Z_STD  = 0.209;
    constexpr double INNER_TUBE_POS_Z_STD  = 4.76e-04;
    constexpr double OUTER_TUBE_ROT_X_STD  = 0.006;
    constexpr double OUTER_TUBE_ROT_Z_STD  = 0.133;

    // Stiffness defaults (override via set_stiffness_params)
    constexpr double OUTER_K_BENDING = 0.00182;
    constexpr double INNER_K_BENDING = 0.00132;
    constexpr double K_TORSION       = 0.0291;

    // Pixel measurement noise
    constexpr double ROBUST_K_PARAMETER = 2.45;
    constexpr double PIXEL_MEAS_STD  = 4.0;

    // Hardware geometry (endoscope + arm cartridge), all in meters/radians.
    //
    // Uterine robot: camera sits BELOW the tools (original Virtuoso design had it above).
    // Tube model follows the robot's own FK in motor_node.cpp: the outer tube is curved over its
    // whole extension (no straight tip, no clearance tilt), the inner tube is straight.
    // Values tagged MEASURE are carried over from the original robot and still need measuring.
    constexpr double OUTER_STRAIGHT_LENGTH     = 0.0;     // was 0.005 (Virtuoso)
    constexpr double NOMINAL_CLEARANCE_ANGLE   = 0.0;     // was 0.19  (Virtuoso)
    constexpr double RETRACTED_CLEARANCE_ANGLE = 0.0;     // was 0.055 (Virtuoso)
    constexpr double INTERARM_DISTANCE         = 0.004; // MEASURE: center-to-center of the two arm channels
    // Sign flipped for camera-below: arms sit on the +y (image-up) side of the camera.
    constexpr double ENDO_BASE_Y_OFFSET        = -0.0025; // MEASURE magnitude (Virtuoso: +0.00324)
    // Sign flipped for camera-below: camera pitches up toward the tools. Use 0 for a forward-looking camera.
    constexpr double CAMERA_ANGLE              = 15.0 * M_PI / 180.0; // MEASURE (Virtuoso: +30 deg)
    constexpr double CAMERA_TO_LENS_Z_OFFSET   = 0.002;
    constexpr double ENDO_INSERTION            = 0.004;   // MEASURE: how far the lens sits beyond the arm channel exits
    // Yaw of each arm base about its own z axis (the tube bends toward base -y at outer_rot = 0).
    // With the camera below the tools:
    //   M_PI: outer_rot = 0 bends the outer tube AWAY from the camera, into view (Virtuoso behavior)
    //   0:    outer_rot = 0 bends it TOWARD the camera
    // Calibration cannot fix a wrong choice: the base yaw prior is ARM_BASE_ROTATION_STD, the limit 45 deg.
    constexpr double ARM_BASE_YAW              = 0.0;     // confirmed: outer_rot = 0 bends toward the camera

    // Geometry-prior uncertainty (used when no calibration is available)
    constexpr double CAMERA_Z_ROTATION_STD     = 20.0 * M_PI / 180.0;
    constexpr double CAMERA_XY_ROTATION_STD    = 5.0  * M_PI / 180.0;
    constexpr double CAMERA_TO_LENS_Z_OFFSET_STD = 0.01;
    constexpr double ENDO_INSERTION_STD        = 0.003;
    constexpr double ARM_BASE_INSERTION_STD    = 0.0005;
    constexpr double ARM_BASE_ROTATION_STD     = 5.0  * M_PI / 180.0;

    // Divide-by-zero guards for near-zero tube segment lengths
    constexpr double CLEARANCE_INTERP_EPSILON = 1e-6;
    constexpr double TORSION_SCALING_EPSILON  = 1e-5;

    // Curvature priors (outer tube is pre-curved; inner is nearly straight)
    // Uterine robot: motor_node.cpp uses CURVATURE_OUTER = 100 1/m, stiffness-weighted with the straight
    // inner tube inside it: 100 * EI_o / (EI_o + EI_i) = 100 * 0.895 = ~90 1/m. (Virtuoso: 50)
    constexpr double OUTER_X_CURVATURE_NOMINAL = 80;
    constexpr double OUTER_X_CURVATURE_PRIOR_STD  = 15.0;
    constexpr double INNER_CURVATURE_PRIOR_STD  = 5.0;

    // Lens pose relative to camera (world): translate in z by camera offset, no rotation
    static const Pose3 LENS_POSE_NOMINAL = Pose3(
        Rot3::Identity(), Point3(0, 0, CAMERA_TO_LENS_Z_OFFSET));

    // Endoscope tip pose relative to lens: rotate by camera angle, no translation
    static const Pose3 LENS_TO_ENDO_TIP_NOMINAL = Pose3(
        Rot3::Rz(M_PI) * Rot3::Rx(-CAMERA_ANGLE), Point3::Zero());

    // Endoscope tip pose relative to endoscope base: rotate to point y axis down, translate to endoscope tip
    static const Pose3 ENDO_TIP_TO_ENDO_BASE_NOMINAL = Pose3(
        Rot3::Identity(), Point3(0, 0, ENDO_INSERTION));

    // Left/Right base poses relative to endo base pose: offset by xy translation (+ ARM_BASE_YAW)
    static const Pose3 LEFT_BASE_TO_ENDO_BASE_NOMINAL = Pose3(
        Rot3::Rz(ARM_BASE_YAW), Point3(INTERARM_DISTANCE / 2, -ENDO_BASE_Y_OFFSET, 0));

    static const Pose3 RIGHT_BASE_TO_ENDO_BASE_NOMINAL = Pose3(
        Rot3::Rz(ARM_BASE_YAW), Point3(-INTERARM_DISTANCE / 2, -ENDO_BASE_Y_OFFSET, 0));

    static const Pose3 ENDO_TIP_POSE_NOMINAL  = LENS_POSE_NOMINAL * LENS_TO_ENDO_TIP_NOMINAL.inverse();
    static const Pose3 ENDO_BASE_POSE_NOMINAL = ENDO_TIP_POSE_NOMINAL * ENDO_TIP_TO_ENDO_BASE_NOMINAL.inverse();
    static const Pose3 LEFT_BASE_POSE_NOMINAL  = ENDO_BASE_POSE_NOMINAL * LEFT_BASE_TO_ENDO_BASE_NOMINAL;
    static const Pose3 RIGHT_BASE_POSE_NOMINAL = ENDO_BASE_POSE_NOMINAL * RIGHT_BASE_TO_ENDO_BASE_NOMINAL;

    static const Vector2 OUTER_CURVATURE_NOMINAL = Vector2(OUTER_X_CURVATURE_NOMINAL, 0);
    static const Vector2 INNER_CURVATURE_NOMINAL = Vector2::Zero();

    // Calibration difference from nominal thresholds
    constexpr double CALIB_MAX_OUTER_CURVATURE_DIFF = 3.0 * OUTER_X_CURVATURE_PRIOR_STD;
    constexpr double CALIB_MAX_INNER_CURVATURE_DIFF = 3.0 * INNER_CURVATURE_PRIOR_STD;
    constexpr double CALIB_MAX_BASE_ROTATION_RAD    = 45.0 * M_PI / 180.0;

    // GTSAM factor graph key helpers

Key base_pose_key(ArmSide side)
{
    return Symbol('B', static_cast<int>(side));
}

Key tube_pose_key(ArmSide side, ArmTube tube, int pose_idx, int time_idx)
{
    char side_char = (side == ArmSide::LEFT) ? 'L' : 'R';
    int tube_idx = static_cast<int>(tube);
    return Symbol(side_char, 1000 * time_idx + 100 * tube_idx + pose_idx);
}

Key error_pose_key(ArmSide side, ArmTube tube, int time_idx)
{
    char side_char = (side == ArmSide::LEFT) ? 'L' : 'R';
    int tube_idx = static_cast<int>(tube);
    // 42 can't collide with valid pose indices 0..(NODES_PER_TUBE-1)
    return Symbol(side_char, 1000 * time_idx + 100 * tube_idx + 42);
}

Key tip_pose_key(ArmSide side, int time_idx)
{
    int side_idx = static_cast<int>(side);
    return Symbol('T', 1000 * time_idx + side_idx);
}

Key curvature_key(ArmSide side, ArmTube tube)
{
    int side_idx = static_cast<int>(side);
    int tube_idx = static_cast<int>(tube);
    return Symbol('U', 100 * side_idx + tube_idx);
}

Key small_curvature_key(ArmSide side)
{
    return Symbol('U', 42 + static_cast<int>(side));
}

Key tip_force_key(ArmSide side, int time_idx)
{
    int side_idx = static_cast<int>(side);
    return Symbol('F', 1000 * time_idx + side_idx);
}

Key endo_tip_pose_key()
{
    return Symbol('E', 42);
}

Key endo_base_pose_key()
{
    return Symbol('E', 43);
}

Key lens_pose_key()
{
    return Symbol('E', 44);
}

} // namespace


SingleArmSample::SingleArmSample(
    double time_seconds,
    const gtsam::Vector4& joint_values,
    const std::vector<gtsam::Vector2>& keypoints,
    const Vector3Gaussian& tip_force,
    const std::optional<Vector3Gaussian>& tip_position_meas)
:
    time_seconds(time_seconds),
    joint_values(joint_values),
    keypoints(keypoints),
    tip_force(tip_force),
    tip_position_meas(tip_position_meas)
{}

SmootherSolver::SmootherSolver(const gtsam::Cal3_S2& camera_intrinsics)
:
    camera_intrinsics_(camera_intrinsics)
{
    init_noise_models();
    set_stiffness_params(OUTER_K_BENDING, INNER_K_BENDING, K_TORSION);
}

void SmootherSolver::init_noise_models()
{
    small_pose_noise_model_ = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector6() <<
        SMALL_ROTATION_STD, SMALL_ROTATION_STD, SMALL_ROTATION_STD,
        SMALL_POSITION_STD, SMALL_POSITION_STD, SMALL_POSITION_STD
    ).finished());

    small_curvature_noise_model_ = noiseModel::Isotropic::Sigma(2, SMALL_CURVATURE_STD);

    set_outer_tube_noise_std(Vector6(
        OUTER_TUBE_ROT_X_STD, SMALL_ROTATION_STD, OUTER_TUBE_ROT_Z_STD,
        SMALL_POSITION_STD, SMALL_POSITION_STD, SMALL_POSITION_STD));

    set_inner_tube_noise_std(Vector6(
        INNER_TUBE_ROT_XY_STD, INNER_TUBE_ROT_XY_STD, INNER_TUBE_ROT_Z_STD,
        SMALL_POSITION_STD, SMALL_POSITION_STD, INNER_TUBE_POS_Z_STD));

    set_pixel_meas_std(PIXEL_MEAS_STD);

    lens_pose_noise_model_ = noiseModel::Diagonal::Sigmas((Vector6() <<
        SMALL_ROTATION_STD, SMALL_ROTATION_STD, SMALL_ROTATION_STD,
        SMALL_POSITION_STD, SMALL_POSITION_STD,
        CAMERA_TO_LENS_Z_OFFSET_STD).finished());

    lens_to_endo_tip_noise_model_ = noiseModel::Diagonal::Sigmas((Vector6() <<
        CAMERA_XY_ROTATION_STD, CAMERA_XY_ROTATION_STD, CAMERA_Z_ROTATION_STD,
        SMALL_POSITION_STD, SMALL_POSITION_STD, SMALL_POSITION_STD).finished());

    endo_tip_to_endo_base_noise_model_ = noiseModel::Diagonal::Sigmas((Vector6() <<
        SMALL_ROTATION_STD, SMALL_ROTATION_STD, SMALL_ROTATION_STD,
        SMALL_POSITION_STD, SMALL_POSITION_STD,
        ENDO_INSERTION_STD).finished());

    arm_base_to_endo_base_noise_model_ = noiseModel::Diagonal::Sigmas((Vector6() <<
        SMALL_ROTATION_STD, SMALL_ROTATION_STD, ARM_BASE_ROTATION_STD,
        SMALL_POSITION_STD, SMALL_POSITION_STD,
        ARM_BASE_INSERTION_STD).finished());

    outer_curvature_prior_noise_model_ = noiseModel::Diagonal::Sigmas(
        (Vector2() << OUTER_X_CURVATURE_PRIOR_STD, SMALL_CURVATURE_STD).finished());

    inner_curvature_prior_noise_model_ = noiseModel::Isotropic::Sigma(2, INNER_CURVATURE_PRIOR_STD);
}

const std::deque<SingleArmSample>& SmootherSolver::samples_for(ArmSide side) const
{
    return (side == ArmSide::LEFT) ? *left_samples_ : *right_samples_;
}

void SmootherSolver::set_tip_accel_prior_std(double accel_prior_std)
{
    tip_accel_prior_noise_model_ = noiseModel::Isotropic::Sigma(3, accel_prior_std);
}

void SmootherSolver::set_pixel_meas_std(double pixel_meas_std)
{
    pixel_noise_model_ = noiseModel::Robust::Create(
        // Cauchy seems to work better than Huber for bad keypoint detections
        // noiseModel::mEstimator::Huber::Create(ROBUST_K_PARAMETER),
        noiseModel::mEstimator::Cauchy::Create(ROBUST_K_PARAMETER), 
        noiseModel::Isotropic::Sigma(2, pixel_meas_std));
}

void SmootherSolver::set_calibration(const SmootherCalibration& calibration)
{
    calibration_ = calibration;
}

void SmootherSolver::set_inner_tube_noise_std(const Vector6& twist_std)
{
    inner_tube_noise_model_ = noiseModel::Diagonal::Sigmas(twist_std);
}

void SmootherSolver::set_outer_tube_noise_std(const Vector6& twist_std)
{
    outer_tube_noise_model_ = noiseModel::Diagonal::Sigmas(twist_std);
}

void SmootherSolver::set_stiffness_params(double outer_k_bending, double inner_k_bending, double k_torsion)
{
    // For K matrices, we assume zero shear and elongation.
    outer_K_inv_ = Matrix6::Zero();
    outer_K_inv_(0, 0) = 1.0 / outer_k_bending;
    outer_K_inv_(1, 1) = 1.0 / outer_k_bending;

    inner_K_inv_ = Matrix6::Zero();
    inner_K_inv_(0, 0) = 1.0 / inner_k_bending;
    inner_K_inv_(1, 1) = 1.0 / inner_k_bending;

    // Torsion is concentrated at the base of the arm (first segment only).
    // torsion_K_inv_(2,2) is stored unscaled — it gets divided by ds_outer_curved
    // in add_single_sample_factors to give the correct per-unit-length compliance.
    torsion_K_inv_ = outer_K_inv_;
    torsion_K_inv_(2, 2) = 1.0 / k_torsion;
}

void SmootherSolver::set_left_tip_offset(const Vector3& offset)
{
    left_tip_offset_ = offset;
}

void SmootherSolver::set_right_tip_offset(const Vector3& offset)
{
    right_tip_offset_ = offset;
}

// Inserts a default value only when the key is absent — existing values are preserved for warm-starting.
// Note this is not the same as Values::insert_or_assign, which would overwrite existing values and break warm-starting.
template<typename T>
static void try_insert(Values& values, Key key, const T& default_val)
{
    if (!values.exists(key)) {
        values.insert(key, default_val);
    }
}

// Inserts default values for all keys belonging to a single arm at a single timestep.
static void init_time_step_values(Values& values, ArmSide side, int time_idx)
{
    try_insert(values, tip_force_key(side, time_idx), Vector3::Zero().eval());
    try_insert(values, tip_pose_key(side, time_idx), Pose3::Identity());

    for (auto tube : {ArmTube::OUTER, ArmTube::INNER}) {
        try_insert(values, error_pose_key(side, tube, time_idx), Pose3::Identity());
        for (int i = 0; i < NODES_PER_TUBE; ++i)
            try_insert(values, tube_pose_key(side, tube, i, time_idx), Pose3::Identity());
    }
}

// All keys referenced by the factor graph must be present in values_ before optimization.
// Uses try_insert to avoid overwriting existing keys so the previous solution warm-starts LM.
void SmootherSolver::init_values()
{
    for (ArmSide side : {ArmSide::LEFT, ArmSide::RIGHT}) {
        const auto& samples = samples_for(side);

        // Note that these here are actually not very good initial values for the solver
        for (size_t t = 0; t < samples.size(); t++)
            init_time_step_values(values_, side, t);

        // However, these remaining calibration values should be good initializations
        try_insert(values_, small_curvature_key(side), Vector2::Zero().eval());
        
        for (auto tube : {ArmTube::OUTER, ArmTube::INNER}) {
            try_insert(values_, curvature_key(side, tube), 
                (tube == ArmTube::OUTER) ? OUTER_CURVATURE_NOMINAL : INNER_CURVATURE_NOMINAL);
        }
        
        try_insert(values_, base_pose_key(side), 
            (side == ArmSide::LEFT) ? LEFT_BASE_POSE_NOMINAL : RIGHT_BASE_POSE_NOMINAL);
    }
}

void SmootherSolver::add_calibration_factors()
{
    // We have to remove these keys if they already exist in the graph, since they won't be optimized in this case.
    // GTSAM will throw an exception for keys that are in Values but not in graph
    if (values_.exists(lens_pose_key()))
        values_.erase(lens_pose_key());

    if (values_.exists(endo_base_pose_key()))
        values_.erase(endo_base_pose_key());

    if (values_.exists(endo_tip_pose_key()))
        values_.erase(endo_tip_pose_key());

    // Ensure we always access the state copy, never the member.
    const auto& c = calibration_.value();
    const auto& l = c.left_arm;
    const auto& r = c.right_arm;

    // Left base pose prior
    graph_.add(PriorFactor<Pose3>(
        base_pose_key(ArmSide::LEFT),
        Pose3(l.base_pose.mean),
        l.base_pose.cov));

    // Right base pose prior
    graph_.add(PriorFactor<Pose3>(
        base_pose_key(ArmSide::RIGHT),
        Pose3(r.base_pose.mean),
        r.base_pose.cov));

    // Left outer tube curvature prior
    graph_.add(PriorFactor<Vector2>(
        curvature_key(ArmSide::LEFT, ArmTube::OUTER),
        l.outer_curvature.mean,
        l.outer_curvature.cov));

    // Left inner tube curvature prior
    graph_.add(PriorFactor<Vector2>(
        curvature_key(ArmSide::LEFT, ArmTube::INNER),
        l.inner_curvature.mean,
        l.inner_curvature.cov));

    // Right outer tube curvature prior
    graph_.add(PriorFactor<Vector2>(
        curvature_key(ArmSide::RIGHT, ArmTube::OUTER),
        r.outer_curvature.mean,
        r.outer_curvature.cov));

    // Right inner tube curvature prior
    graph_.add(PriorFactor<Vector2>(
        curvature_key(ArmSide::RIGHT, ArmTube::INNER),
        r.inner_curvature.mean,
        r.inner_curvature.cov));
}

void SmootherSolver::add_single_sample_factors(ArmSide side, int time_idx, bool include_keypoints)
{
    const auto& sample = samples_for(side)[time_idx];

    // Tip force prior
    graph_.add(PriorFactor<Vector3>(
        tip_force_key(side, time_idx),
        sample.tip_force.mean,
        sample.tip_force.cov));

    // Kinematics of the robot
    double inner_z_rotation = sample.joint_values[0];
    double outer_z_rotation = sample.joint_values[1];
    double inner_tube_extension = std::max(0.0, sample.joint_values[2]);
    double outer_tube_extension = std::max(0.0, sample.joint_values[3]);

    double outer_rod_straight_length = std::min(outer_tube_extension, OUTER_STRAIGHT_LENGTH);
    double outer_rod_curved_length   = outer_tube_extension - outer_rod_straight_length;
    double inner_rod_length = std::max(0.0, (inner_tube_extension - outer_tube_extension));

    // First linearly interpolate clearance angle based on how much of the outer tube straight section is extended
    double clearance_angle = RETRACTED_CLEARANCE_ANGLE +
        (NOMINAL_CLEARANCE_ANGLE - RETRACTED_CLEARANCE_ANGLE) *
        (outer_rod_straight_length / (OUTER_STRAIGHT_LENGTH + CLEARANCE_INTERP_EPSILON));

    // Pose offsets for each rod base rotation
    Pose3 inner_rot_pose_mean = Pose3(Rot3::Rz(inner_z_rotation - outer_z_rotation), Point3::Zero());
    Pose3 outer_rot_pose_mean = Pose3(Rot3::Rz(outer_z_rotation) * Rot3::Rx(clearance_angle), Point3::Zero());

    // Rotate outer error pose to nominal tube base pose
    graph_.add(BetweenFactor<Pose3>(
        base_pose_key(side),
        error_pose_key(side, ArmTube::OUTER, time_idx),
        outer_rot_pose_mean,
        small_pose_noise_model_));

    // Prior for tube base relative to error pose, inject error pose noise
    graph_.add(BetweenFactor<Pose3>(
        error_pose_key(side, ArmTube::OUTER, time_idx),
        tube_pose_key(side, ArmTube::OUTER, 0, time_idx),
        Pose3::Identity(),
        outer_tube_noise_model_));

    // Inject inner tube error at the tip of outer tube, in the outer tube frame (important)
    graph_.add(BetweenFactor<Pose3>(
        tube_pose_key(side, ArmTube::OUTER, NODES_PER_TUBE - 1, time_idx),
        error_pose_key(side, ArmTube::INNER, time_idx),
        Pose3::Identity(),
        inner_tube_noise_model_));

    // Rotate and translate inner tube
    graph_.add(BetweenFactor<Pose3>(
        error_pose_key(side, ArmTube::INNER, time_idx),
        tube_pose_key(side, ArmTube::INNER, 0, time_idx),
        inner_rot_pose_mean,
        small_pose_noise_model_));

    // Cosserat twist factors for outer tube segments
    double ds_outer_curved = outer_rod_curved_length / (NODES_PER_TUBE - 2); // -2 since straight length is separate
    double ds_outer_straight = outer_rod_straight_length;
    double ds_inner = inner_rod_length / (NODES_PER_TUBE - 1);

    // All torsion is modeled as occurring at the base, so modify the first segment's K_inv to reflect this

    Matrix6 torsion_K_inv_scaled = torsion_K_inv_;
    torsion_K_inv_scaled(2, 2) /= (ds_outer_curved + TORSION_SCALING_EPSILON);

    // Outer tube cosserat factors complicated by torsion and straight section
    for (int i = 0; i + 1 < NODES_PER_TUBE; ++i) {
        const bool     curved_seg = (i < NODES_PER_TUBE - 2);

        // We concentrate all torsion only at the base segment, since physically it is twisting inside the endoscope
        const Matrix6& K_inv      = (i == 0) ? torsion_K_inv_scaled : outer_K_inv_;

        // The staight section at then end of the tube has different spacing and zero curvature 
        const double   ds         = curved_seg ? ds_outer_curved   : ds_outer_straight;
        const Key      curv_key   = curved_seg ? curvature_key(side, ArmTube::OUTER) : small_curvature_key(side);

        graph_.add(CosseratTwistFactor(
            tube_pose_key(side, ArmTube::OUTER, i, time_idx),
            tube_pose_key(side, ArmTube::OUTER, i + 1, time_idx),
            tip_pose_key(side, time_idx), tip_force_key(side, time_idx),
            curv_key, ds, K_inv, small_pose_noise_model_));
    }

    // Inner tube cosserat factors are much simpler
    for (int i = 0; i + 1 < NODES_PER_TUBE; ++i) {
        graph_.add(CosseratTwistFactor(
            tube_pose_key(side, ArmTube::INNER, i, time_idx),
            tube_pose_key(side, ArmTube::INNER, i + 1, time_idx),
            tip_pose_key(side, time_idx), tip_force_key(side, time_idx),
            curvature_key(side, ArmTube::INNER),
            ds_inner, inner_K_inv_, small_pose_noise_model_));
    }

    // Tool z offset between last inner tube pose and the tip
    auto& tip_offset = (side == ArmSide::LEFT) ? left_tip_offset_ : right_tip_offset_;

    graph_.add(BetweenFactor<Pose3>(
        tube_pose_key(side, ArmTube::INNER, NODES_PER_TUBE - 1, time_idx),
        tip_pose_key(side, time_idx),
        Pose3(Rot3::Identity(), tip_offset),
        small_pose_noise_model_));

    // Single factor over all keypoints using max mixture approximation for data association.
    // Intentionally uses the inner tube last node (not tip_pose_key): the keypoint model was
    // trained on the bare tube tip, and tip_offset accounts for whatever tool is mounted there.
    // 
    // Note: skipped in stage 1 of a 2 stage solve so geometry initialises before keypoints pull on it
    if (include_keypoints && !sample.keypoints.empty()) {
        graph_.add(TipProjectionFactor(
            tube_pose_key(side, ArmTube::INNER, NODES_PER_TUBE - 1, time_idx),
            sample.keypoints,
            camera_intrinsics_,
            pixel_noise_model_));
    }

    // Optional absolute tip position constraint (e.g., from an external tracking system)
    if (sample.tip_position_meas) {
        auto noise = noiseModel::Gaussian::Covariance(sample.tip_position_meas->cov);

        graph_.add(PositionPriorFactor(
            tube_pose_key(side, ArmTube::INNER, NODES_PER_TUBE - 1, time_idx),
            sample.tip_position_meas->mean,
            noise));
    }
}


void SmootherSolver::add_geometry_prior_factors()
{
    // Endo/lens keys are only in the graph in geometry-prior mode; insert defaults preserving warm starts.
    try_insert(values_, lens_pose_key(), LENS_POSE_NOMINAL);
    try_insert(values_, endo_base_pose_key(), ENDO_BASE_POSE_NOMINAL);
    try_insert(values_, endo_tip_pose_key(), ENDO_TIP_POSE_NOMINAL);

    // Lens pose relative to camera
    graph_.add(PriorFactor<Pose3>(
        lens_pose_key(),
        LENS_POSE_NOMINAL,
        lens_pose_noise_model_));
    
    // Lens pose relative to endoscope tip
    graph_.add(BetweenFactor<Pose3>(
        endo_tip_pose_key(),
        lens_pose_key(),
        LENS_TO_ENDO_TIP_NOMINAL,
        lens_to_endo_tip_noise_model_));
    
    // Endoscope tip pose relative to endoscope base
    graph_.add(BetweenFactor<Pose3>(
        endo_base_pose_key(),
        endo_tip_pose_key(),
        ENDO_TIP_TO_ENDO_BASE_NOMINAL,
        endo_tip_to_endo_base_noise_model_));
    
    // Left and Right base poses relative to endoscope base
    graph_.add(BetweenFactor<Pose3>(
        endo_base_pose_key(),
        base_pose_key(ArmSide::LEFT),
        LEFT_BASE_TO_ENDO_BASE_NOMINAL,
        arm_base_to_endo_base_noise_model_));

    graph_.add(BetweenFactor<Pose3>(
        endo_base_pose_key(),
        base_pose_key(ArmSide::RIGHT),
        RIGHT_BASE_TO_ENDO_BASE_NOMINAL,
        arm_base_to_endo_base_noise_model_));

    // Left and Right curvature priors
    //   Outer tube:
    //     x: assumed curved (~OUTER_X_CURVATURE_NOMINAL) but with some prior uncertainty.
    //     y: is assumed to be certainly small
    //  Inner tube
    //    x and y: assumed to be small, but with some prior uncertainty

    for (auto arm_side : {ArmSide::LEFT, ArmSide::RIGHT}) {
        graph_.add(PriorFactor<Vector2>(
            curvature_key(arm_side, ArmTube::OUTER),
            OUTER_CURVATURE_NOMINAL,
            outer_curvature_prior_noise_model_));

        graph_.add(PriorFactor<Vector2>(
            curvature_key(arm_side, ArmTube::INNER),
            INNER_CURVATURE_NOMINAL,
            inner_curvature_prior_noise_model_));
    }
}

void SmootherSolver::add_motion_prior_factors()
{
    // Only add these factors if the prior motion model is enabled, since sometimes (calibration) we don't want it
    if (!tip_accel_prior_noise_model_) {
        return;
    }

    for (ArmSide side : {ArmSide::LEFT, ArmSide::RIGHT}) {
        const auto& arm_samples = samples_for(side);

        for (size_t t = 2; t < arm_samples.size(); t++) {
            double t1 = arm_samples[t - 2].time_seconds;
            double t2 = arm_samples[t - 1].time_seconds;
            double t3 = arm_samples[t].time_seconds;

            // Skip if any consecutive pair shares a timestamp — would cause division by zero in ConstantVelocityFactor
            if (t2 <= t1 || t3 <= t2) continue;

            graph_.add(ConstantVelocityFactor(
                tip_pose_key(side, t - 2),
                tip_pose_key(side, t - 1),
                tip_pose_key(side, t),
                t1, t2, t3,
                tip_accel_prior_noise_model_));
        }
    }
}

void SmootherSolver::build_graph(bool include_keypoints)
{
    // New graph each update
    graph_ = NonlinearFactorGraph();

    // Small curvature keys used for straight sections
    graph_.add(PriorFactor<Vector2>(
        small_curvature_key(ArmSide::LEFT),
        Vector2::Zero(),
        small_curvature_noise_model_));

    graph_.add(PriorFactor<Vector2>(
        small_curvature_key(ArmSide::RIGHT),
        Vector2::Zero(),
        small_curvature_noise_model_));

    // If we have a calibration, use it, else use the prior geometry.
    // The calibration already contains the prior information so we don't add it twice.
    if (calibration_) {
        add_calibration_factors();
    } else {
        add_geometry_prior_factors();
    }

    for (auto side : {ArmSide::LEFT, ArmSide::RIGHT}) {
        for (size_t t = 0; t < samples_for(side).size(); t++) {
            add_single_sample_factors(side, t, include_keypoints);
        }
    }

    add_motion_prior_factors();
}

template <typename Rep, typename Period>
double milliseconds(const std::chrono::duration<Rep, Period>& duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

void extract_jac_position_joints(
    SingleArmMarginals& solution,
    const Pose3& base_pose,
    double x_curvature)
{
    Pose3 outer_tube_base_pose = Pose3(solution.outer_tube_poses[0].mean);
    Pose3 inner_tube_base_pose = Pose3(solution.inner_tube_poses[0].mean);
    Pose3 tip_pose = Pose3(solution.tip_pose.mean);

    Vector3 p = tip_pose.translation() - base_pose.translation();
    Vector3 tip_pose_z_axis = tip_pose.rotation().r3();
    Vector3 base_z_axis = base_pose.rotation().r3();
    Vector3 inner_tube_base_z_axis = inner_tube_base_pose.rotation().r3();
    Vector3 outer_tube_base_x_axis = outer_tube_base_pose.rotation().r1();
    Vector3 outer_tube_base_z_axis = outer_tube_base_pose.rotation().r3();

    // Inner rotation does not affect tip position
    Matrix34 Jv;
    Jv.col(0) = Vector3::Zero();

    // Outer rotation moves tip position in a circle, use cross product
    Jv.col(1) = cross(base_z_axis, p);

    // Inner translation moves the tip in the direction of the tip pose z axis
    Jv.col(2) = tip_pose_z_axis;

    // Outer translation is the most complicated, since there are weird effects between inner/outer tubes.
    // First we get the velocity of the outer tip wrt base. Note this approximate, since "curvature" will
    // be different when we're in the straight section.
    Vector3 omega_x = x_curvature * outer_tube_base_x_axis;
    Jv.col(3) = outer_tube_base_z_axis - inner_tube_base_z_axis + cross(omega_x, p);

    // Rotations, much easier now that we have positions
    Matrix34 Jw;
    Jw.col(0) = inner_tube_base_z_axis;
    Jw.col(1) = base_z_axis - inner_tube_base_z_axis;
    Jw.col(2) = Vector3::Zero();
    Jw.col(3) = omega_x;

    solution.jac_tip_pose.topRows<3>() = Jw;
    solution.jac_tip_pose.bottomRows<3>() = Jv;
}

void compute_joint_uvz(
    const Pose3Gaussian& tip_pose,
    const Cal3_S2& camera_intrinsics,
    Vector3Gaussian& uvz)
{
    Matrix36 d_uvz_d_tip_pose;
    bool behind_camera = project_pose_to_uvz(
        Pose3(tip_pose.mean),
        camera_intrinsics,
        uvz.mean,
        d_uvz_d_tip_pose);

    if (!behind_camera) {
        uvz.cov = d_uvz_d_tip_pose * tip_pose.cov * d_uvz_d_tip_pose.transpose();
    } else {
        uvz.mean.setConstant(std::numeric_limits<double>::quiet_NaN());
        uvz.cov = Eigen::Matrix3d::Identity();
    }
}

static Pose3Gaussian extract_pose_gaussian(Key key, const Values& v, const Marginals& m)
{
    return {v.at<Pose3>(key).matrix(), m.marginalCovariance(key)};
}

static Vector2Gaussian extract_vector2_gaussian(Key key, const Values& v, const Marginals& m)
{
    return {v.at<Vector2>(key), m.marginalCovariance(key)};
}

static Vector3Gaussian extract_vector3_gaussian(Key key, const Values& v, const Marginals& m)
{
    return {v.at<Vector3>(key), m.marginalCovariance(key)};
}

void SmootherSolver::extract_single_arm(
    ArmSide side,
    int time_idx,
    SingleArmMarginals& solution)
{
    const auto& v = values_;
    const auto& m = marginals_;

    // Extract all tube pose marginals
    for (auto tube : {ArmTube::OUTER, ArmTube::INNER}) {
        auto& tube_poses = (tube == ArmTube::OUTER)
            ? solution.outer_tube_poses
            : solution.inner_tube_poses;

        auto& tube_uvz = (tube == ArmTube::OUTER)
            ? solution.outer_tube_uvz
            : solution.inner_tube_uvz;

        for (int i = 0; i < NODES_PER_TUBE; ++i) {
            auto key = tube_pose_key(side, tube, i, time_idx);
            tube_poses[i] = extract_pose_gaussian(key, v, m);
            compute_joint_uvz(tube_poses[i], camera_intrinsics_, tube_uvz[i]);
        }
    }

    // Extract tip marginals
    solution.tip_pose  = extract_pose_gaussian(tip_pose_key(side, time_idx),  v, m);
    solution.tip_force = extract_vector3_gaussian(tip_force_key(side, time_idx), v, m);

    // Tip pixel prediction
    compute_joint_uvz(solution.tip_pose, camera_intrinsics_, solution.tip_uvz);

    // Note curvature is approximate, does not include effects of clearance angles
    double x_curvature = v.at<Vector2>(curvature_key(side, ArmTube::OUTER))[0];
    Pose3 base_pose = v.at<Pose3>(base_pose_key(side));
    extract_jac_position_joints(solution, base_pose, x_curvature);
}

static void check_single_arm_calibration(SingleArmCalibration& calib, const Pose3& nominal)
{
    calib.outer_curvature_diff = calib.outer_curvature.mean - OUTER_CURVATURE_NOMINAL;
    calib.inner_curvature_diff = calib.inner_curvature.mean - INNER_CURVATURE_NOMINAL;
    calib.base_pose_diff = Pose3::Logmap(nominal.between(Pose3(calib.base_pose.mean)));

    calib.is_valid =
        calib.outer_curvature_diff.norm() < CALIB_MAX_OUTER_CURVATURE_DIFF &&
        calib.inner_curvature_diff.norm() < CALIB_MAX_INNER_CURVATURE_DIFF &&
        calib.base_pose_diff.head<3>().norm() < CALIB_MAX_BASE_ROTATION_RAD;
}

void SmootherSolver::extract_calibration(SmootherSolution& solution)
{
    const auto& v = values_;
    const auto& m = marginals_;
    auto& l = solution.calibration.left_arm;
    auto& r = solution.calibration.right_arm;

    l.base_pose       = extract_pose_gaussian(base_pose_key(ArmSide::LEFT),  v, m);
    r.base_pose       = extract_pose_gaussian(base_pose_key(ArmSide::RIGHT), v, m);
    l.outer_curvature = extract_vector2_gaussian(curvature_key(ArmSide::LEFT,  ArmTube::OUTER), v, m);
    l.inner_curvature = extract_vector2_gaussian(curvature_key(ArmSide::LEFT,  ArmTube::INNER), v, m);
    r.outer_curvature = extract_vector2_gaussian(curvature_key(ArmSide::RIGHT, ArmTube::OUTER), v, m);
    r.inner_curvature = extract_vector2_gaussian(curvature_key(ArmSide::RIGHT, ArmTube::INNER), v, m);

    // Meta: calibrator (no calibration_ set) counts its own accumulated samples;
    // tracker (calibration_ set) preserves the count from the calibrator so the saved
    // file reflects how many samples went into the calibration, not the tracker window.
    l.num_samples = calibration_ ? calibration_->left_arm.num_samples  : samples_for(ArmSide::LEFT).size();
    r.num_samples = calibration_ ? calibration_->right_arm.num_samples : samples_for(ArmSide::RIGHT).size();
    solution.calibration.total_time_ms = solution.total_time_ms;

    // Tracker solves have fixed calibration priors and never use is_valid.
    if (!calibration_) {
        check_single_arm_calibration(solution.calibration.left_arm,  LEFT_BASE_POSE_NOMINAL);
        check_single_arm_calibration(solution.calibration.right_arm, RIGHT_BASE_POSE_NOMINAL);
        solution.calibration.is_valid = l.is_valid && r.is_valid;
    }
}

void SmootherSolver::extract_solution(SmootherSolution& solution)
{
    for (auto side : {ArmSide::LEFT, ArmSide::RIGHT}) {
        auto& arm_solution = (side == ArmSide::LEFT) ? solution.left_arm : solution.right_arm;
        for (size_t t = 0; t < arm_solution.size(); t++) {
            extract_single_arm(side, t, arm_solution[t]);
        }
    }
    extract_calibration(solution);
}

SmootherSolution SmootherSolver::solve(
    const std::deque<SingleArmSample>& left_samples,
    const std::deque<SingleArmSample>& right_samples,
    int max_iterations,
    bool two_stage)
{
    left_samples_  = &left_samples;
    right_samples_ = &right_samples;

    // Values carried over from last solve for warm start; inserts nominals for any new keys.
    init_values();

    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();

    DoglegParams params;
    params.setMaxIterations(max_iterations);

    // Stage 1 (optional): solve without keypoint factors so the geometry settles to a
    // physically consistent state before keypoints are introduced. This prevents the
    // TipProjectionFactor from latching onto wrong pixels when the initial tube-tip
    // positions are far from their true locations.
    if (two_stage) {
        build_graph(/*include_keypoints=*/false);
        values_ = DoglegOptimizer(graph_, values_, params).optimize();
    }

    // Stage 2 (or only stage): full solve including keypoint factors.
    build_graph(/*include_keypoints=*/true);
    auto stop_build = Clock::now();

    DoglegOptimizer optimizer(graph_, values_, params);
    values_ = optimizer.optimize();
    auto stop_optimize = Clock::now();

    // Extract marginals and solution
    marginals_ = Marginals(graph_, values_);

    SmootherSolution solution;
    solution.left_arm.resize(left_samples_->size());
    solution.right_arm.resize(right_samples_->size());
    extract_solution(solution);
    auto stop_extract = Clock::now();

    // build_time_ms covers everything before the final optimize (includes stage-1 if used)
    solution.build_time_ms    = milliseconds(stop_build    - start);
    solution.optimize_time_ms = milliseconds(stop_optimize - stop_build);
    solution.extract_time_ms  = milliseconds(stop_extract  - stop_optimize);
    solution.total_time_ms    = milliseconds(stop_extract  - start);

    solution.iterations = optimizer.iterations();
    solution.error      = optimizer.error();

    left_samples_  = nullptr;
    right_samples_ = nullptr;

    return solution;
}
