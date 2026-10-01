#include "solver/SmootherSolver.h"

#include "HyperCalibrator.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#define BIND_FIELD(cls, field) def_readwrite(#field, &cls::field)


PYBIND11_MODULE(smoother_uterus, m) {

    py::class_<Vector2Gaussian>(m, "Vector2Gaussian")
        .def(py::init<>())
        .def(py::init<const gtsam::Vector2&, const gtsam::Matrix2&>(),
            py::arg("mean"), py::arg("cov"))
        .BIND_FIELD(Vector2Gaussian, mean)
        .BIND_FIELD(Vector2Gaussian, cov);

    py::class_<Vector3Gaussian>(m, "Vector3Gaussian")
        .def(py::init<>())
        .def(py::init<const gtsam::Vector3&, const gtsam::Matrix3&>(),
            py::arg("mean"), py::arg("cov"))
        .BIND_FIELD(Vector3Gaussian, mean)
        .BIND_FIELD(Vector3Gaussian, cov);

    py::class_<Pose3Gaussian>(m, "Pose3Gaussian")
        .def(py::init<>())
        .def(py::init<const gtsam::Matrix4&, const gtsam::Matrix6&>(),
            py::arg("mean"), py::arg("cov"))
        .BIND_FIELD(Pose3Gaussian, mean)
        .BIND_FIELD(Pose3Gaussian, cov);

    py::class_<SingleArmCalibration>(m, "SingleArmCalibration")
        .def(py::init<>())
        .BIND_FIELD(SingleArmCalibration, base_pose)
        .BIND_FIELD(SingleArmCalibration, outer_curvature)
        .BIND_FIELD(SingleArmCalibration, inner_curvature)
        .BIND_FIELD(SingleArmCalibration, num_samples);

    py::class_<SmootherCalibration>(m, "SmootherCalibration")
        .def(py::init<>())
        .BIND_FIELD(SmootherCalibration, left_arm)
        .BIND_FIELD(SmootherCalibration, right_arm)
        .BIND_FIELD(SmootherCalibration, total_time_ms);

    py::class_<SingleArmMarginals>(m, "SingleArmMarginals")
        .def(py::init<>())
        .BIND_FIELD(SingleArmMarginals, outer_tube_poses)
        .BIND_FIELD(SingleArmMarginals, inner_tube_poses)
        .BIND_FIELD(SingleArmMarginals, tip_pose)
        .BIND_FIELD(SingleArmMarginals, tip_force)
        .BIND_FIELD(SingleArmMarginals, jac_tip_pose)
        .BIND_FIELD(SingleArmMarginals, outer_tube_uvz)
        .BIND_FIELD(SingleArmMarginals, inner_tube_uvz)
        .BIND_FIELD(SingleArmMarginals, tip_uvz);

    py::class_<SmootherSolution>(m, "SmootherSolution")
        .def(py::init<>())
        .BIND_FIELD(SmootherSolution, left_arm)
        .BIND_FIELD(SmootherSolution, right_arm)
        .BIND_FIELD(SmootherSolution, calibration)
        .BIND_FIELD(SmootherSolution, error)
        .BIND_FIELD(SmootherSolution, iterations)
        .BIND_FIELD(SmootherSolution, build_time_ms)
        .BIND_FIELD(SmootherSolution, optimize_time_ms)
        .BIND_FIELD(SmootherSolution, extract_time_ms)
        .BIND_FIELD(SmootherSolution, total_time_ms);

    py::class_<SingleArmSample>(m, "SingleArmSample")
        .def(py::init<>())
        .BIND_FIELD(SingleArmSample, time_seconds)
        .BIND_FIELD(SingleArmSample, joint_values)
        .BIND_FIELD(SingleArmSample, tip_force)
        .BIND_FIELD(SingleArmSample, keypoints)
        .BIND_FIELD(SingleArmSample, tip_position_meas);

    // Camera intrinsics are irrelevant for Python dev/simulation use (no real camera).
    // solve() takes Python lists which pybind11/stl.h auto-converts to
    // std::deque<SingleArmSample>. Deque window management lives in ves_solver.py.
    py::class_<SmootherSolver>(m, "SmootherSolver")
        .def(py::init([]() {
            return new SmootherSolver(gtsam::Cal3_S2(800.0, 800.0, 0.0, 320.0, 240.0));
        }))
        .def("solve", &SmootherSolver::solve,
            py::arg("left_samples"), py::arg("right_samples"),
            py::call_guard<py::gil_scoped_release>())
        .def("set_calibration",          &SmootherSolver::set_calibration)
        .def("set_inner_tube_noise_std", &SmootherSolver::set_inner_tube_noise_std)
        .def("set_outer_tube_noise_std", &SmootherSolver::set_outer_tube_noise_std)
        .def("set_stiffness_params",     &SmootherSolver::set_stiffness_params)
        .def("set_tip_accel_prior_std",  &SmootherSolver::set_tip_accel_prior_std)
        .def("set_left_tip_offset",      &SmootherSolver::set_left_tip_offset)
        .def("set_right_tip_offset",     &SmootherSolver::set_right_tip_offset)
        .def("set_pixel_meas_std",       &SmootherSolver::set_pixel_meas_std);

    py::class_<HyperCalibratorSolution>(m, "HyperCalibratorSolution")
        .def(py::init<>())
        .def_readwrite("pose_errors",              &HyperCalibratorSolution::pose_errors)
        .def_readwrite("whitened_pose_errors",     &HyperCalibratorSolution::whitened_pose_errors)
        .def_readwrite("position_errors",          &HyperCalibratorSolution::position_errors)
        .def_readwrite("whitened_position_errors", &HyperCalibratorSolution::whitened_position_errors)
        .def_readwrite("nll",                      &HyperCalibratorSolution::nll);

    py::class_<HyperCalibrator>(m, "HyperCalibrator")
        .def(py::init<std::vector<gtsam::Vector4>&>())
        .def("solve", &HyperCalibrator::solve,
            py::arg("tip_pose_meas"),
            py::arg("tip_position_meas"),
            py::arg("tip_force_meas"),
            py::arg("outer_curvature"),
            py::arg("inner_curvature"),
            py::arg("inner_tube_noise_std"),
            py::arg("outer_tube_noise_std"),
            py::arg("outer_k_bending"),
            py::arg("inner_k_bending"),
            py::arg("k_torsion"),
            py::call_guard<py::gil_scoped_release>());
}
