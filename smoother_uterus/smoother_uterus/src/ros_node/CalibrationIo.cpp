#include "CalibrationIo.h"

#include <chrono>
#include <ctime>
#include <fstream>

#include <yaml-cpp/node/node.h>

namespace {

// Version 2: base poses and camera mount in the endoscope tip frame (smoother_uterus/endo), valid at any view
// angle. Version 1 (no version field) stored base poses in the camera frame of one fixed view angle.
constexpr int CALIBRATION_FORMAT_VERSION = 2;
constexpr const char* CALIBRATION_FRAME = "smoother_uterus/endo";

YAML::Node encode_eigen_yaml(const Eigen::MatrixXd& mat) 
{
    YAML::Node node;
    if (mat.cols() == 1) {
        // Vector -> flat list
        for (int i = 0; i < mat.rows(); ++i)
            node.push_back(mat(i, 0));
        node.SetStyle(YAML::EmitterStyle::Flow);
    } else {
        // Matrix -> list of rows
        for (int i = 0; i < mat.rows(); ++i) {
            YAML::Node row;
            for (int j = 0; j < mat.cols(); ++j)
                row.push_back(mat(i, j));
            row.SetStyle(YAML::EmitterStyle::Flow);
            node.push_back(row);
        }
    }
    return node;
}

template <int Size>
Eigen::Vector<double, Size> decode_vector_yaml(const YAML::Node& node) 
{
    if (node.size() != Size) {
        std::stringstream ss;
        ss << "YAML data size (" << node.size() << ") does not match expected size (" << Size << ").";
        throw std::runtime_error(ss.str());
    }
    Eigen::Vector<double, Size> v;
    for (size_t i = 0; i < Size; ++i) {
        v[i] = node[i].as<double>();
    }
    return v;
}

template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> decode_matrix_yaml(const YAML::Node& node) 
{
    Eigen::Matrix<double, Rows, Cols> m;
    if (node.size() != Rows) {
        std::stringstream ss;
        ss << "YAML row size (" << node.size() << ") does not match expected row size (" << Rows << ").";
        throw std::runtime_error(ss.str());
    }
    for (size_t i = 0; i < node.size(); ++i) {
        if (node[i].size() != Cols) {
            std::stringstream ss;
            ss << "YAML column size (" << node[i].size() << ") does not match expected column size (" << Cols << ").";
            throw std::runtime_error(ss.str());
        }

        for (size_t j = 0; j < node[i].size(); ++j) {
            m(i, j) = node[i][j].as<double>();
        }
    }
    return m;
}

int64_t get_unix_seconds()
{
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

void save_arm_calibration(YAML::Node arm_node, const SingleArmCalibration& arm)
{
    for (auto tube : {ArmTube::OUTER, ArmTube::INNER}) {
        const char* tube_label = (tube == ArmTube::OUTER) ? "outer" : "inner";
        YAML::Node tube_node = arm_node["tubes"][tube_label];
        const auto& t = (tube == ArmTube::OUTER) ? arm.outer_curvature : arm.inner_curvature;
        tube_node["curvature_mean"] = encode_eigen_yaml(t.mean);
        tube_node["curvature_cov"]  = encode_eigen_yaml(t.cov);
    }
    arm_node["base_pose_mean"] = encode_eigen_yaml(arm.base_pose.mean.matrix());
    arm_node["base_pose_cov"]  = encode_eigen_yaml(arm.base_pose.cov);
    arm_node["num_samples"]    = arm.num_samples;
}

void load_arm_calibration(const YAML::Node& arm_node, SingleArmCalibration& arm)
{
    for (auto tube : {ArmTube::OUTER, ArmTube::INNER}) {
        const char* tube_label = (tube == ArmTube::OUTER) ? "outer" : "inner";
        const YAML::Node& tube_node = arm_node["tubes"][tube_label];
        auto& t = (tube == ArmTube::OUTER) ? arm.outer_curvature : arm.inner_curvature;
        t.mean = decode_vector_yaml<2>(tube_node["curvature_mean"]);
        t.cov  = decode_matrix_yaml<2, 2>(tube_node["curvature_cov"]);
    }
    arm.base_pose.mean = decode_matrix_yaml<4, 4>(arm_node["base_pose_mean"]);
    arm.base_pose.cov  = decode_matrix_yaml<6, 6>(arm_node["base_pose_cov"]);
    if (arm_node["num_samples"])
        arm.num_samples = arm_node["num_samples"].as<size_t>();
}

} // namespace

void save_calibration_yaml(const SmootherCalibration& c, const std::string& filename)
{
    YAML::Node node;

    node["format_version"] = CALIBRATION_FORMAT_VERSION;
    node["frame"] = CALIBRATION_FRAME;

    save_arm_calibration(node["arms"]["left"],  c.left_arm);
    save_arm_calibration(node["arms"]["right"], c.right_arm);

    if (c.camera_mount) {
        node["camera_mount"]["pose_mean"] = encode_eigen_yaml(c.camera_mount->mean.matrix());
        node["camera_mount"]["pose_cov"]  = encode_eigen_yaml(c.camera_mount->cov);
    }

    node["total_time_ms"] = c.total_time_ms;

    // When was this calibration performed?
    node["timestamp_seconds"] = get_unix_seconds();

    // Throw and halt if it can't open or write calibration
    try {
        std::ofstream fout;
        fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        fout.open(filename);

        fout << node;
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("Failed to write calibration YAML to: '" + filename + "': " + e.what());
    }
}

LoadedCalibration load_calibration_yaml(const std::string& filename)
{
    // This will throw if the file doesn't exist or is malformed, which is what we want, since we tried to load a calibration
    YAML::Node node = YAML::LoadFile(filename);

    // Older files hold base poses in the camera frame of a single view angle; loading them now would put the
    // arms in the wrong place, so refuse instead.
    const int version = node["format_version"] ? node["format_version"].as<int>() : 1;
    if (version < CALIBRATION_FORMAT_VERSION) {
        throw std::runtime_error(
            "Calibration '" + filename + "' was saved before the moving-camera change (base poses in the camera "
            "frame). Recalibrate to make one in the endoscope frame.");
    }

    SmootherCalibration c;

    load_arm_calibration(node["arms"]["left"],  c.left_arm);
    load_arm_calibration(node["arms"]["right"], c.right_arm);

    if (node["camera_mount"]) {
        Pose3Gaussian mount;
        mount.mean = decode_matrix_yaml<4, 4>(node["camera_mount"]["pose_mean"]);
        mount.cov  = decode_matrix_yaml<6, 6>(node["camera_mount"]["pose_cov"]);
        c.camera_mount = mount;
    }

    if (node["total_time_ms"]) {
        c.total_time_ms = node["total_time_ms"].as<double>();
    }

    LoadedCalibration loaded;
    loaded.calibration = c;

    // TODO remove after testing. But right now I am using a dataset that doesnt have this field
    if (node["timestamp_seconds"]){
        loaded.age_seconds = get_unix_seconds() - node["timestamp_seconds"].as<int64_t>();
    }

    return loaded;
}
