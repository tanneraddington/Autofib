#include "SolutionPublisher.h"

#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Point2.h>

using namespace gtsam;

using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::PoseWithCovarianceStamped;
using geometry_msgs::msg::TransformStamped;
using visualization_msgs::msg::MarkerArray;
using visualization_msgs::msg::Marker;
using sensor_msgs::msg::Image;
using std_msgs::msg::Float64MultiArray;

namespace {
    const std::string TOPIC_NS = "smoother_uterus";

    // Topics are published in the camera frame of their image (FRAME_ID), as before the view angle could
    // change. TF and rviz markers use the endoscope tip frame (ENDO_FRAME_ID), the solver's world frame, which
    // stays put when the view swings. TF tree: endo -> camera (moves with the view angle), endo -> <side>/base,
    // endo -> <side>/tip.
    const std::string FRAME_ID      = TOPIC_NS + "/camera";
    const std::string ENDO_FRAME_ID = TOPIC_NS + "/endo";

    const std::array<std::string, 2> arm_labels = {"left", "right"};
    const std::array<std::string, 2> tube_labels = {"outer", "inner"};

    // Tube visualization geometry (cylinder markers)
    constexpr double OUTER_TUBE_MARKER_DIAMETER = 0.0008;
    constexpr double INNER_TUBE_MARKER_DIAMETER = 0.0007;

    // Force arrow visualization geometry
    constexpr double FORCE_ARROW_SHAFT_DIAMETER = 0.00025;
    constexpr double FORCE_ARROW_HEAD_DIAMETER  = 0.0005;
    constexpr double FORCE_ARROW_HEAD_LENGTH    = 0.0005;

    // Camera field-of-view marker: pyramid from the camera out to this depth (m)
    constexpr double FOV_MARKER_DEPTH      = 0.04;
    constexpr double FOV_MARKER_LINE_WIDTH = 0.0002;

    // Minimum force magnitude to display markers (suppresses noise visualization)
    constexpr double FORCE_DISPLAY_THRESHOLD = 1e-3;

    // Force/uncertainty marker color (purple)
    constexpr float FORCE_COLOR_R = 102.0f / 255.0f;
    constexpr float FORCE_COLOR_G = 51.0f  / 255.0f;
    constexpr float FORCE_COLOR_B = 153.0f / 255.0f;

    // Overlay text style
    constexpr int    TEXT_FONT        = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double TEXT_SCALE       = 1.0;
    constexpr int    TEXT_WEIGHT      = 2;
    constexpr int    TEXT_MARGIN      = 10;
    constexpr int    TEXT_LINE_HEIGHT = 35;
}

void draw_covariance_ellipse(
    cv::Mat& image,
    const cv::Point& center,
    const Eigen::Matrix2d& cov,
    int line_width,
    double scale = 2.0)
{
    // Check if covariance is finite
    if (!cov.allFinite())
        return;

    // Make symmetric
    Eigen::Matrix2d cov_sym = 0.5 * (cov + cov.transpose());

    // Solve for eigenvectors and eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov_sym);
    Eigen::Vector2d eig_vals = eig.eigenvalues();
    Eigen::Matrix2d eig_vecs = eig.eigenvectors();

    // Identify major and minor axes
    int major_idx = eig_vals(1) > eig_vals(0) ? 1 : 0;
    int minor_idx = 1 - major_idx;
    double major = scale * std::sqrt(std::max(eig_vals(major_idx), 0.0));
    double minor = scale * std::sqrt(std::max(eig_vals(minor_idx), 0.0));

    // Convert to pixels, ensure at least 1 pixel
    int major_px = std::max(1, static_cast<int>(std::lround(major)));
    int minor_px = std::max(1, static_cast<int>(std::lround(minor)));

    // Get major axis to compute ellipse angle
    Eigen::Vector2d major_axis = eig_vecs.col(major_idx);

    // Compute angle from x-axis, clockwise
    double angle_rad = std::atan2(major_axis(1), major_axis(0));
    double angle_deg = angle_rad * 180.0 / M_PI;

    // Normalize angle to [0, 360)
    angle_deg = std::fmod(angle_deg + 360.0, 360.0);

    // Ensure axes, and angle are finite
    if (!std::isfinite(major) || !std::isfinite(minor) || !std::isfinite(angle_deg))
        return;

    // Ensure the axes are greater than zero
    if (major <= 0 || minor <= 0)
        return;

    cv::ellipse(image, center, cv::Size(major_px, minor_px), angle_deg,
        0, 360, cv::Scalar(0, 215, 255), line_width, cv::LINE_AA);
}

void draw_backbone(
    cv::Mat& image,
    const SingleArmMarginals& m)
{
    for (const auto& tube : {m.outer_tube_uvz, m.inner_tube_uvz}) {
        std::optional<cv::Point2d> last_point;
        for (const auto& uvz : tube) {
            if (uvz.mean.z() <= 1e-6) continue;

            if (!std::isfinite(uvz.mean.x()) || !std::isfinite(uvz.mean.y()) || !std::isfinite(uvz.mean.z()))
                continue;

            cv::Point2d point(uvz.mean.x(), uvz.mean.y());

            if (last_point) {
                int line_width = 2;
                cv::line(image, *last_point, point, cv::Scalar(200, 0, 0), line_width, cv::LINE_AA);
            }

            last_point = point;
        }
    }
}

void draw_geometries(
    cv::Mat& image,
    const SingleArmMarginals& m,
    const std::vector<Point2>& keypoints,
    const cv::Scalar& color)
{
    // Draw all keypoints as crosses
    for (const auto& keypoint : keypoints) {
        cv::Point2d uv_meas = cv::Point2d(keypoint.x(), keypoint.y());
        cv::drawMarker(image, uv_meas, color, cv::MARKER_TILTED_CROSS, 25, 2, cv::LINE_AA);
    }

    double x = m.tip_uvz.mean.x();
    double y = m.tip_uvz.mean.y();

    // Check if its finite
    if (!std::isfinite(x) || !std::isfinite(y))
        return;

    cv::Point2d tip_uv_mean = cv::Point2d(m.tip_uvz.mean.x(), m.tip_uvz.mean.y());
    Matrix2 tip_uv_cov = m.tip_uvz.cov.block<2,2>(0,0);

    cv::circle(image, tip_uv_mean, 5, color, -1, cv::LINE_AA);
    draw_covariance_ellipse(image, tip_uv_mean, tip_uv_cov, 2);
}

double rms_error(const Matrix& cov)
{
    return std::sqrt(cov.trace());
}

cv::Scalar arm_color(ArmSide side) {
    return (side == ArmSide::LEFT) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
}

// Renders a column of text lines anchored to either the top or bottom edge of the image.
// Lines stack away from the anchor edge (bottom lines go upward, top lines go downward).
static void draw_text_column(
    cv::Mat& image,
    const std::vector<std::string>& lines,
    ArmSide side,
    bool from_bottom,
    const cv::Scalar& color)
{
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::Size sz = cv::getTextSize(lines[i], TEXT_FONT, TEXT_SCALE, TEXT_WEIGHT, nullptr);
        int x = (side == ArmSide::LEFT) ? TEXT_MARGIN : image.cols - TEXT_MARGIN - sz.width;
        int y = from_bottom
            ? image.rows - TEXT_MARGIN - static_cast<int>(i) * TEXT_LINE_HEIGHT
            : TEXT_MARGIN + sz.height + static_cast<int>(i) * TEXT_LINE_HEIGHT;
        cv::putText(image, lines[i], {x, y}, TEXT_FONT, TEXT_SCALE, color, TEXT_WEIGHT, cv::LINE_AA);
    }
}

void draw_text(
    cv::Mat& image,
    const SingleArmMarginals& m,
    const SmootherCalibration& calibration,
    bool received_stop_calibration,
    const std::vector<Point2>& keypoints,
    ArmSide side)
{
    // Bottom: keypoint count, depth, and arm label
    std::ostringstream depth_label;
    depth_label << std::fixed << std::setw(4) << std::setprecision(1) << m.tip_uvz.mean.z() * 1000.0;

    std::vector<std::string> bottom_lines = {
        "KEYPOINTS: " + std::to_string(keypoints.size()),
        "DEPTH: " + depth_label.str(),
        (side == ArmSide::LEFT) ? "LEFT" : "RIGHT"};

    draw_text_column(image, bottom_lines, side, true, arm_color(side));

    // Top: calibration uncertainty, or locked-in indicator
    std::vector<std::string> top_lines;

    const auto& base_pose = (side == ArmSide::LEFT) ? calibration.left_arm.base_pose : calibration.right_arm.base_pose;

    if (received_stop_calibration) {
        top_lines.push_back("LOCKED IN");
    } else {
        std::ostringstream rot_label;
        rot_label << "Rot (deg): " << std::fixed << std::setw(5) << std::setprecision(1)
            << rms_error(base_pose.cov.block<3,3>(0,0)) * (180.0 / M_PI);
        top_lines.push_back(rot_label.str());

        std::ostringstream pos_label;
        pos_label << "Pos  (mm): " << std::fixed << std::setw(5) << std::setprecision(1)
            << rms_error(base_pose.cov.block<3,3>(3,3)) * 1000.0;
        top_lines.push_back(pos_label.str());
    }

    draw_text_column(image, top_lines, side, false, cv::Scalar(255, 255, 255));
}

cv::Mat make_overlay_image(const SolvedPipelineSample& solved)
{
    cv::Mat image = solved.keypoint.raw.image.clone();

    const auto& left     = solved.solution.left_arm.back();
    const auto& right    = solved.solution.right_arm.back();
    const auto& left_kp  = solved.keypoint.keypoints.left_keypoints;
    const auto& right_kp = solved.keypoint.keypoints.right_keypoints;
    const auto& calib    = solved.solution.calibration;

    draw_backbone(image, left);
    draw_backbone(image, right);
    draw_geometries(image, left,  left_kp,  arm_color(ArmSide::LEFT));
    draw_geometries(image, right, right_kp, arm_color(ArmSide::RIGHT));
    draw_text(image, left,  calib, solved.calibration_stopped, left_kp,  ArmSide::LEFT);
    draw_text(image, right, calib, solved.calibration_stopped, right_kp, ArmSide::RIGHT);

    return image;
}

// Endoscope-frame quantities -> this image's camera frame. GTSAM pose covariances are in body coordinates,
// which a change of reference frame leaves unchanged; vectors and Jacobians rotate.
Pose3Gaussian to_camera(const Pose3Gaussian& pose, const Pose3& camera_from_endo)
{
    return {(camera_from_endo * Pose3(pose.mean)).matrix(), pose.cov};
}

std::array<Pose3Gaussian, NODES_PER_TUBE> to_camera(
    const std::array<Pose3Gaussian, NODES_PER_TUBE>& poses, const Pose3& camera_from_endo)
{
    std::array<Pose3Gaussian, NODES_PER_TUBE> out;
    for (size_t i = 0; i < poses.size(); ++i)
        out[i] = to_camera(poses[i], camera_from_endo);
    return out;
}

Vector3Gaussian rotate_to_camera(const Vector3Gaussian& v, const Matrix3& R)
{
    return {R * v.mean, R * v.cov * R.transpose()};
}

Matrix64 jacobian_to_camera(const Matrix64& J, const Matrix3& R)
{
    Matrix64 out;
    out.topRows<3>()    = R * J.topRows<3>();     // angular velocity per joint
    out.bottomRows<3>() = R * J.bottomRows<3>();  // tip linear velocity per joint
    return out;
}

Pose gtsam_pose_to_msg(const Matrix4& pose)
{
    Pose msg;

    const auto& t = pose.block<3,1>(0,3);
    msg.position.x = t.x();
    msg.position.y = t.y();
    msg.position.z = t.z();

    const auto& R = pose.block<3,3>(0,0);
    Eigen::Quaterniond q(R);
    msg.orientation.x = q.x();
    msg.orientation.y = q.y();
    msg.orientation.z = q.z();
    msg.orientation.w = q.w();

    return msg;
}

void publish_pose_array(
    const std::array<Pose3Gaussian, NODES_PER_TUBE>& poses,
    const rclcpp::Publisher<PoseArray>::SharedPtr& pub,
    const rclcpp::Time& stamp)
{
    PoseArray msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = FRAME_ID;

    for (const auto& pose : poses)
        msg.poses.push_back(gtsam_pose_to_msg(pose.mean));

    pub->publish(msg);
}

void publish_pose_with_cov(
    const Pose3Gaussian& pose,
    const rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr& pub,
    const rclcpp::Time& stamp)
{
    PoseWithCovarianceStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = FRAME_ID;

    msg.pose.pose = gtsam_pose_to_msg(pose.mean);

    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            msg.pose.covariance[r * 6 + c] = pose.cov(r, c);

    pub->publish(msg);
}

void publish_force_with_cov(
    const Vector3Gaussian& force,
    const rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr& pub,
    const rclcpp::Time& stamp)
{
    // Abusing PoseWithCovarianceStamped to publish force with covariance since there's no Vector3WithCovariance
    PoseWithCovarianceStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = FRAME_ID;

    // Position field stores the force vector, orientation field is unused
    msg.pose.pose.position.x = force.mean.x();
    msg.pose.pose.position.y = force.mean.y();
    msg.pose.pose.position.z = force.mean.z();

    // Fill top left 3x3 block of covariance with force covariance; rest is unused
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            msg.pose.covariance[r * 6 + c] = force.cov(r, c);

    pub->publish(msg);
}

void publish_jac_tip_pose(
    const Matrix64& jac_tip_pose,
    const rclcpp::Publisher<Float64MultiArray>::SharedPtr& pub,
    const rclcpp::Time& stamp)
{
    Float64MultiArray msg;
    msg.data.resize(jac_tip_pose.rows() * jac_tip_pose.cols());

    // Row-major packing
    for (int r = 0; r < jac_tip_pose.rows(); ++r)
        for (int c = 0; c < jac_tip_pose.cols(); ++c)
            msg.data[r * jac_tip_pose.cols() + c] = jac_tip_pose(r, c);

    pub->publish(msg);
}

void send_pose_as_tf(
    const Matrix4& pose,
    tf2_ros::TransformBroadcaster* broadcaster,
    const std::string& child_frame_id,
    const rclcpp::Time& stamp,
    const std::string& parent_frame_id)
{
    geometry_msgs::msg::Pose msg = gtsam_pose_to_msg(pose);

    TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = parent_frame_id;
    t.child_frame_id = child_frame_id;
    t.transform.translation.x = msg.position.x;
    t.transform.translation.y = msg.position.y;
    t.transform.translation.z = msg.position.z;
    t.transform.rotation = msg.orientation;

    broadcaster->sendTransform(t);
}

void publish_pixels(
    const Vector3Gaussian& uvz,
    const rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr& pub,
    const rclcpp::Time& stamp)
{
    PoseWithCovarianceStamped msg;
    msg.header.stamp = stamp;

    msg.pose.pose.position.x = uvz.mean.x(); // image x
    msg.pose.pose.position.y = uvz.mean.y(); // image y
    msg.pose.pose.position.z = uvz.mean.z(); // depth z

    msg.pose.covariance[0] = uvz.cov(0,0);
    msg.pose.covariance[1] = uvz.cov(0,1);
    msg.pose.covariance[2] = uvz.cov(0,2);
    msg.pose.covariance[3] = uvz.cov(1,0);
    msg.pose.covariance[4] = uvz.cov(1,1);
    msg.pose.covariance[5] = uvz.cov(1,2);
    msg.pose.covariance[6] = uvz.cov(2,0);
    msg.pose.covariance[7] = uvz.cov(2,1);
    msg.pose.covariance[8] = uvz.cov(2,2);

    pub->publish(msg);
}

void publish_tube_marker_array(
    const std::array<Pose3Gaussian, NODES_PER_TUBE>& poses,
    const std::string& ns,
    ArmTube tube,
    const rclcpp::Publisher<MarkerArray>::SharedPtr& pub,
    const rclcpp::Time& stamp,
    const std::string& frame_id)
{
    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.reserve(NODES_PER_TUBE - 1);
    int marker_id = 0;

    for (int i = 0; i < NODES_PER_TUBE - 1; ++i) {
        const auto& p1 = poses[i].mean.block<3,1>(0,3);
        const auto& p2 = poses[i + 1].mean.block<3,1>(0,3);

        Vector3 dir = p2 - p1;
        double height = dir.norm();

        Vector3 midpoint = 0.5 * (p1 + p2);
        Eigen::Quaterniond q = (height < 1e-5)
            ? Eigen::Quaterniond::Identity()
            : Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), dir.normalized());

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = stamp;
        marker.ns = ns + "_tube";
        marker.id = marker_id++;
        marker.type = marker.CYLINDER;
        marker.action = marker.ADD;
        marker.pose.position.x = midpoint.x();
        marker.pose.position.y = midpoint.y();
        marker.pose.position.z = midpoint.z();
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        marker.scale.z = height;
        marker.lifetime = rclcpp::Duration::from_seconds(0.5);

        marker.color.a = (height < 1e-5) ? 0.0f : 1.0f; // Hide marker if its very thin

        if (tube == ArmTube::OUTER) {
            marker.scale.x = marker.scale.y = OUTER_TUBE_MARKER_DIAMETER;
            marker.color.r = 0.2f;
            marker.color.g = 0.25f;
            marker.color.b = 0.7f;
        } else {
            marker.scale.x = marker.scale.y = INNER_TUBE_MARKER_DIAMETER;
            marker.color.r = 0.2f;
            marker.color.g = 0.5f;
            marker.color.b = 0.9f;
        }

        marker_array.markers.push_back(marker);
    }

    pub->publish(marker_array);
}

Marker get_uncertainty_ellipsoid_marker(const Vector3& p, const Matrix3& p_cov, const double scale_factor = 1.0) {
    Marker marker;
    marker.type = Marker::SPHERE;
    marker.action = Marker::ADD;

    // Eigen-decompose symmetric PSD; get sorted axes
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(p_cov);
    Eigen::Vector3d eigenvals = solver.eigenvalues();     // ascending
    Eigen::Matrix3d eigenvecs = solver.eigenvectors();    // columns

    // Sort by descending eigenvalue (X >= Y >= Z)
    std::array<int, 3> order = {2, 1, 0}; // eigenvalues from solver are ascending
    Eigen::Matrix3d U;      // eigenvectors in columns
    U.col(0) = eigenvecs.col(order[0]);
    U.col(1) = eigenvecs.col(order[1]);
    U.col(2) = eigenvecs.col(order[2]);
    Eigen::Vector3d lam(eigenvals[order[0]], eigenvals[order[1]], eigenvals[order[2]]);

    // Ensure a right-handed basis (det +1)
    if (U.determinant() < 0.0) U.col(2) *= -1.0;

    // Radii for 2-sigma ellipsoid; marker scale expects DIAMETERS
    constexpr double k_sigma = 2.0;
    Eigen::Vector3d radii = k_sigma * lam.cwiseMax(0.0).cwiseSqrt();
    Eigen::Vector3d diam = 2.0 * scale_factor * radii; // Scale factor to make ellipsoid scales match between position and force visualizations

    Eigen::Quaterniond q_marker(U);

    marker.pose.position.x = p.x();
    marker.pose.position.y = p.y();
    marker.pose.position.z = p.z();

    marker.pose.orientation.x = q_marker.x();
    marker.pose.orientation.y = q_marker.y();
    marker.pose.orientation.z = q_marker.z();
    marker.pose.orientation.w = q_marker.w();

    const double min_vis = 1e-6;
    marker.scale.x = std::max(diam.x(), min_vis);
    marker.scale.y = std::max(diam.y(), min_vis);
    marker.scale.z = std::max(diam.z(), min_vis);

    return marker;
}

void publish_tip_position_uncertainty_marker(
    const Pose3Gaussian& tip_pose,
    const rclcpp::Publisher<Marker>::SharedPtr& pub,
    const rclcpp::Time& stamp,
    const std::string& frame_id)
{
    // Unpack pose and covariance to eigen matrices
    Point3 p = tip_pose.mean.block<3,1>(0,3);
    Matrix3 R = tip_pose.mean.block<3,3>(0,0);

    Matrix3 p_cov_body = tip_pose.cov.block<3,3>(3,3);
    p_cov_body = 0.5 * (p_cov_body + p_cov_body.transpose()); // Numerical symmetrization (help tiny asymmetries)
    Matrix3 p_cov = R * p_cov_body * R.transpose(); // Rotate to WORLD for visualization

    Marker marker = get_uncertainty_ellipsoid_marker(p, p_cov);

    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns   = "tip_ellipsoid";
    marker.id   = 0;

    marker.color.r = 1.0; marker.color.g = 0.85; marker.color.b = 0.0; marker.color.a = 0.6;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    pub->publish(marker);
}

void publish_tip_force_markers(
    const SingleArmMarginals& arm,
    const rclcpp::Publisher<Marker>::SharedPtr& arrow_pub,
    const rclcpp::Publisher<Marker>::SharedPtr& uncertainty_pub,
    const rclcpp::Time& stamp,
    const std::string& frame_id)
{
    Vector3 f_mean = arm.tip_force.mean;
    Matrix3 f_cov = arm.tip_force.cov;
    Vector3 p = arm.tip_pose.mean.block<3,1>(0,3);

    // Scale factor to make force arrows visible in the same scale as position uncertainty ellipsoids
    // They look giant without scaling, since forces are on the order of 0.1N and positions are on the order of 0.001m
    const double force_vis_scale = 0.01;
    Vector3 f_scaled = force_vis_scale * f_mean;

    // Only publish if the mean force is above a small threshold to avoid visualizing noise
    if (f_mean.norm() < FORCE_DISPLAY_THRESHOLD) {
        // Delete any previously published markers so they don't stick in RViz
        Marker del;
        del.header.frame_id = frame_id;
        del.header.stamp = stamp;
        del.action = Marker::DELETE;
        del.ns = "tip_force_arrow"; del.id = 0;
        arrow_pub->publish(del);
        del.ns = "tip_force_uncertainty";
        uncertainty_pub->publish(del);
        return;
    }

    // Arrow marker for mean force
    Marker arrow_marker;
    arrow_marker.header.frame_id = frame_id;
    arrow_marker.header.stamp = stamp;
    arrow_marker.ns = "tip_force_arrow";
    arrow_marker.id = 0;
    arrow_marker.type = Marker::ARROW;
    arrow_marker.action = Marker::ADD;
    arrow_marker.points.resize(2);
    arrow_marker.points[0].x = p.x();
    arrow_marker.points[0].y = p.y();
    arrow_marker.points[0].z = p.z();
    arrow_marker.points[1].x = p.x() + f_scaled.x();
    arrow_marker.points[1].y = p.y() + f_scaled.y();
    arrow_marker.points[1].z = p.z() + f_scaled.z();
    arrow_marker.scale.x = FORCE_ARROW_SHAFT_DIAMETER;
    arrow_marker.scale.y = FORCE_ARROW_HEAD_DIAMETER;
    arrow_marker.scale.z = FORCE_ARROW_HEAD_LENGTH;
    arrow_marker.color.r = FORCE_COLOR_R;
    arrow_marker.color.g = FORCE_COLOR_G;
    arrow_marker.color.b = FORCE_COLOR_B;
    arrow_marker.color.a = 0.8f;
    arrow_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    arrow_pub->publish(arrow_marker);

    // Ellipsoid marker for force uncertainty
    Marker uncertainty_marker = get_uncertainty_ellipsoid_marker(p + f_scaled, f_cov, force_vis_scale);

    uncertainty_marker.header.frame_id = frame_id;
    uncertainty_marker.header.stamp = stamp;
    uncertainty_marker.ns = "tip_force_uncertainty";
    uncertainty_marker.id = 0;
    uncertainty_marker.color.r = FORCE_COLOR_R;
    uncertainty_marker.color.g = FORCE_COLOR_G;
    uncertainty_marker.color.b = FORCE_COLOR_B;
    uncertainty_marker.color.a = 0.3f;
    uncertainty_marker.lifetime = rclcpp::Duration::from_seconds(0.1); // Short lifetime to avoid clutter, since force uncertainty can be large and visually overwhelming

    uncertainty_pub->publish(uncertainty_marker);
}

SolutionPublisher::SolutionPublisher(rclcpp::Node& node) : node_(node) {
    for (auto side : {ArmSide::LEFT, ArmSide::RIGHT}) {
        auto arm_label = arm_labels[static_cast<int>(side)];

        for (auto tube : {ArmTube::OUTER, ArmTube::INNER}) {
            auto tube_label = tube_labels[static_cast<int>(tube)];

            // Pose array pubs
            pose_array_pubs_[static_cast<int>(side)][static_cast<int>(tube)] = node_.create_publisher<PoseArray>(
                TOPIC_NS + "/" + arm_label + "/" + tube_label + "/poses", 10);

            // Marker array pubs
            tube_marker_pubs_[static_cast<int>(side)][static_cast<int>(tube)] = node_.create_publisher<MarkerArray>(
                TOPIC_NS + "/" + arm_label + "/" + tube_label + "/markers", 10);
        }

        // Base and tip pose pubs
        tip_pose_pubs_[static_cast<int>(side)] = node_.create_publisher<PoseWithCovarianceStamped>(
            TOPIC_NS + "/" + arm_label + "/tip_pose", 10);
        base_pose_pubs_[static_cast<int>(side)] = node_.create_publisher<PoseWithCovarianceStamped>(
            TOPIC_NS + "/" + arm_label + "/base_pose", 10);

        // Tip forces with covariance pubs
        tip_force_pubs_[static_cast<int>(side)] = node_.create_publisher<PoseWithCovarianceStamped>(
            TOPIC_NS + "/" + arm_label + "/tip_force", 10);

        // Jacobians relating tip position (camera frame) to joint values
        jac_position_joints_pubs_[static_cast<int>(side)] = node_.create_publisher<Float64MultiArray>(
            TOPIC_NS + "/" + arm_label + "/jac_tip_pose", 10);

        // TF broadcasters (parent: endoscope frame)
        base_tf_broadcasters_[static_cast<int>(side)] = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
        tip_tf_broadcasters_[static_cast<int>(side)] = std::make_shared<tf2_ros::TransformBroadcaster>(node_);

        // Predicted pixel location pubs
        pixels_pubs_[static_cast<int>(side)] = node_.create_publisher<PoseWithCovarianceStamped>(
            TOPIC_NS + "/" + arm_label + "/tool_tip_pixels", 10);

        // Tip position uncertainty marker pubs
        tip_position_uncertainty_marker_pubs_[static_cast<int>(side)] = node_.create_publisher<Marker>(
            TOPIC_NS + "/" + arm_label + "/tip_position_uncertainty_marker", 10);

        // Tip force arrow and uncertainty marker pubs
        tip_force_arrow_marker_pubs_[static_cast<int>(side)] = node_.create_publisher<Marker>(
            TOPIC_NS + "/" + arm_label + "/tip_force_arrow_marker", 10);
        tip_force_uncertainty_marker_pubs_[static_cast<int>(side)] = node_.create_publisher<Marker>(
            TOPIC_NS + "/" + arm_label + "/tip_force_uncertainty_marker", 10);
    }

    // Image overlay pub
    overlay_image_pub_ = node_.create_publisher<Image>(
        TOPIC_NS + "/overlay_image", 10);

    // Endoscope -> camera TF, which changes with the view angle
    camera_tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);

    // Camera field of view (rviz)
    camera_fov_pub_ = node_.create_publisher<MarkerArray>(TOPIC_NS + "/camera_fov", 10);
}

void SolutionPublisher::set_camera_intrinsics(const gtsam::Cal3_S2& camera_intrinsics)
{
    camera_intrinsics_ = camera_intrinsics;
}

void SolutionPublisher::publish_single_arm(
        const SolvedPipelineSample& solved,
        const ArmSide side) const
{
    // Use the most recent solution in the lag window for publishing
    const auto& arm_vec = (side == ArmSide::LEFT) ? solved.solution.left_arm : solved.solution.right_arm;
    if (arm_vec.empty()) return;
    const auto& arm = arm_vec.back();
    const auto& stamp = solved.keypoint.raw.stamp;
    const int s = static_cast<int>(side);
    const auto& side_str = arm_labels[s];
    const auto& calib_arm = (side == ArmSide::LEFT) ? solved.solution.calibration.left_arm : solved.solution.calibration.right_arm;

    // The solver works in the endoscope frame; this image's camera frame depends on its view angle
    const Pose3 camera_from_endo(arm.camera_from_endo);
    const Matrix3 R_camera_from_endo = camera_from_endo.rotation().matrix();

    // Topics, in this image's camera frame (what the visual servo and other consumers expect)
    publish_pose_array(to_camera(arm.outer_tube_poses, camera_from_endo), pose_array_pubs_[s][static_cast<int>(ArmTube::OUTER)], stamp);
    publish_pose_array(to_camera(arm.inner_tube_poses, camera_from_endo), pose_array_pubs_[s][static_cast<int>(ArmTube::INNER)], stamp);
    publish_pixels(arm.tip_uvz, pixels_pubs_[s], stamp);
    publish_pose_with_cov(to_camera(arm.tip_pose, camera_from_endo), tip_pose_pubs_[s], stamp);
    publish_jac_tip_pose(jacobian_to_camera(arm.jac_tip_pose, R_camera_from_endo), jac_position_joints_pubs_[s], stamp);
    publish_force_with_cov(rotate_to_camera(arm.tip_force, R_camera_from_endo), tip_force_pubs_[s], stamp);
    publish_pose_with_cov(to_camera(calib_arm.base_pose, camera_from_endo), base_pose_pubs_[s], stamp);

    // TF, in the endoscope frame
    send_pose_as_tf(arm.tip_pose.mean, tip_tf_broadcasters_[s].get(), TOPIC_NS + "/" + side_str + "/tip", stamp, ENDO_FRAME_ID);
    send_pose_as_tf(calib_arm.base_pose.mean, base_tf_broadcasters_[s].get(), TOPIC_NS + "/" + side_str + "/base", stamp, ENDO_FRAME_ID);

    // Visualization markers, in the endoscope frame (they stay put when the view swings)
    publish_tube_marker_array(arm.outer_tube_poses, side_str + "_outer", ArmTube::OUTER, tube_marker_pubs_[s][static_cast<int>(ArmTube::OUTER)], stamp, ENDO_FRAME_ID);
    publish_tube_marker_array(arm.inner_tube_poses, side_str + "_inner", ArmTube::INNER, tube_marker_pubs_[s][static_cast<int>(ArmTube::INNER)], stamp, ENDO_FRAME_ID);
    publish_tip_position_uncertainty_marker(arm.tip_pose, tip_position_uncertainty_marker_pubs_[s], stamp, ENDO_FRAME_ID);
    publish_tip_force_markers(arm, tip_force_arrow_marker_pubs_[s], tip_force_uncertainty_marker_pubs_[s], stamp, ENDO_FRAME_ID);
}

void SolutionPublisher::publish_camera_tf(const SolvedPipelineSample& solved) const
{
    // Both arms' samples for an image share its camera
    const auto& arm_vec = solved.solution.left_arm.empty() ? solved.solution.right_arm : solved.solution.left_arm;
    if (arm_vec.empty()) return;

    const Pose3 camera_in_endo = Pose3(arm_vec.back().camera_from_endo).inverse();
    send_pose_as_tf(camera_in_endo.matrix(), camera_tf_broadcaster_.get(), FRAME_ID, solved.keypoint.raw.stamp, ENDO_FRAME_ID);
}

void SolutionPublisher::publish_overlay_image(const SolvedPipelineSample& solved) const
{
    cv::Mat overlay_image = make_overlay_image(solved);

    auto overlay_image_msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), "bgr8", overlay_image).toImageMsg();

    overlay_image_msg->header.stamp = solved.keypoint.raw.stamp;
    overlay_image_msg->header.frame_id = FRAME_ID;

    overlay_image_pub_->publish(*overlay_image_msg);
}

void SolutionPublisher::publish_camera_fov(const SolvedPipelineSample& solved) const
{
    if (!camera_intrinsics_)
        return;
    const auto& K = *camera_intrinsics_;
    const double width  = solved.keypoint.raw.image.cols;
    const double height = solved.keypoint.raw.image.rows;
    if (width <= 0 || height <= 0)
        return;

    // Image corners back-projected to FOV_MARKER_DEPTH, in the camera frame (x right, y down, z forward)
    auto corner = [&](double u, double v) {
        geometry_msgs::msg::Point p;
        p.x = (u - K.px()) / K.fx() * FOV_MARKER_DEPTH;
        p.y = (v - K.py()) / K.fy() * FOV_MARKER_DEPTH;
        p.z = FOV_MARKER_DEPTH;
        return p;
    };
    const geometry_msgs::msg::Point apex;  // camera centre
    const std::array<geometry_msgs::msg::Point, 4> corners = {
        corner(0, 0), corner(width, 0), corner(width, height), corner(0, height)};

    Marker edges;
    edges.header.frame_id = FRAME_ID;
    edges.header.stamp = solved.keypoint.raw.stamp;
    edges.ns = "camera_fov_edges";
    edges.id = 0;
    edges.type = Marker::LINE_LIST;
    edges.action = Marker::ADD;
    edges.pose.orientation.w = 1.0;
    edges.scale.x = FOV_MARKER_LINE_WIDTH;
    edges.color.r = 0.2f; edges.color.g = 0.6f; edges.color.b = 1.0f; edges.color.a = 0.9f;
    edges.lifetime = rclcpp::Duration::from_seconds(0.5);

    Marker faces = edges;
    faces.ns = "camera_fov_faces";
    faces.type = Marker::TRIANGLE_LIST;
    faces.scale.x = faces.scale.y = faces.scale.z = 1.0;
    faces.color.a = 0.08f;

    for (size_t i = 0; i < corners.size(); ++i) {
        const auto& a = corners[i];
        const auto& b = corners[(i + 1) % corners.size()];
        edges.points.insert(edges.points.end(), {apex, a, a, b});   // edge from the camera, then the far rim
        faces.points.insert(faces.points.end(), {apex, a, b});      // one side of the pyramid
    }

    MarkerArray markers;
    markers.markers = {edges, faces};
    camera_fov_pub_->publish(markers);
}

void SolutionPublisher::publish(const SolvedPipelineSample& solved) const
{
    publish_camera_tf(solved);
    publish_camera_fov(solved);
    publish_single_arm(solved, ArmSide::LEFT);
    publish_single_arm(solved, ArmSide::RIGHT);
    publish_overlay_image(solved);
}
