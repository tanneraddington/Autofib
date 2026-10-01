import numpy as np
from scipy.spatial.transform import Rotation

import smoother_uterus


class VesSingleArmSolver:
    def __init__(self, window_size=1):
        self.solver = smoother_uterus.SmootherSolver()
        self.window_size = window_size
        self.left_samples = []

        # Default calibration on construction
        self.set_calibration()

        # For warm starting inverse kinematics
        self.joint_state = np.array([0.0, 0.0, 0.03, 0.015])  # Don't start at singularity

    def set_calibration(self,
                        outer_curvature_mean=np.array((90.0, 0)),  # uterine robot, see SmootherSolver.cpp
                        outer_curvature_cov=1e-2 ** 2 * np.eye(2),
                        inner_curvature_mean=np.zeros(2),
                        inner_curvature_cov=1e-2 ** 2 * np.eye(2),
                        base_pose_mean=np.eye(4),
                        base_pose_cov=1e-8 * np.eye(6)):

        c = smoother_uterus.SmootherCalibration()

        c.left_arm.outer_curvature.mean = outer_curvature_mean
        c.left_arm.outer_curvature.cov = outer_curvature_cov
        c.left_arm.inner_curvature.mean = inner_curvature_mean
        c.left_arm.inner_curvature.cov = inner_curvature_cov
        c.left_arm.base_pose.mean = base_pose_mean
        c.left_arm.base_pose.cov = base_pose_cov

        # Not using the right arm, just set it to be the same
        c.right_arm = c.left_arm

        self.solver.set_calibration(c)

    def set_noise(self, inner_tube_noise_std=None, outer_tube_noise_std=None):
        if inner_tube_noise_std is not None:
            self.solver.set_inner_tube_noise_std(inner_tube_noise_std)

        if outer_tube_noise_std is not None:
            self.solver.set_outer_tube_noise_std(outer_tube_noise_std)

    def forward_kinematics(self,
                           q,
                           f=np.zeros(3),
                           f_cov=1.0e-3 ** 2 * np.eye(3),
                           p_meas=None,
                           p_meas_cov=1.3e-4 ** 2 * np.eye(3)):

        arm = smoother_uterus.SingleArmSample()
        arm.time_seconds = 0.0
        arm.joint_values = q
        arm.tip_force = smoother_uterus.Vector3Gaussian(f, f_cov)

        if p_meas is not None:
            arm.tip_position_meas = smoother_uterus.Vector3Gaussian(p_meas, p_meas_cov)

        self.left_samples.append(arm)
        if len(self.left_samples) > self.window_size:
            self.left_samples.pop(0)

        return self.solver.solve(self.left_samples, [])

    def get_dummy_solution(self):
        return self.forward_kinematics(self.joint_state)

    def inverse_kinematics(self, goal_pose, f=np.zeros(3), max_iter=100, print_iter=False):
        p_goal = goal_pose[:3, 3]
        R_goal = goal_pose[:3, :3]

        for iter in range(max_iter):
            # Solve given current joint values, force
            solution = self.forward_kinematics(self.joint_state, f)
            pose = solution.left_arm[0].tip_pose.mean
            J = solution.left_arm[0].jac_tip_pose

            # Define errors, jacobians
            ep = p_goal - pose[:3, 3]
            er = Rotation.from_matrix(R_goal @ pose[:3, :3].T).as_rotvec()
            Jp = J[3:, :]
            Jr = J[:3, :]

            # Position gets task priority
            lam = 1e-5
            Hp = Jp.T @ Jp + lam * np.eye(Jp.shape[1])
            gp = Jp.T @ ep
            dq_position = np.linalg.solve(Hp, gp)

            # Use svd to get the nullspace of position jacobian
            U, S, Vt = np.linalg.svd(Jp)
            rank = np.sum(S > 1e-8)
            Z = Vt.T[:, rank:]  # nullspace basis

            # Solve for rotation dq, project onto nullspace
            rhs = er - Jr @ dq_position
            A = Jr @ Z
            y = np.linalg.pinv(A) @ rhs
            dq_rotation = Z @ y

            # Final update, printing, stopping criteria
            dq = dq_position + dq_rotation
            self.joint_state += dq

            lower = np.array([-np.inf, -np.inf, 0.0, 0.0])
            upper = np.array([np.inf, np.inf, 0.06, 0.04])  # uterine robot: inner 60 mm, outer 40 mm
            self.joint_state = np.clip(self.joint_state, lower, upper)

            ep_norm = np.linalg.norm(ep)
            er_norm = np.linalg.norm(er)
            dq_norm = np.linalg.norm(dq)

            if print_iter:
                print(f"iter {iter}: ||ep||={ep_norm:.6f}, ||er||={er_norm:.6f}, ||dq||={dq_norm:.6f}")

            # Stop if joints are not changing and ep is small
            if dq_norm < 1e-6 and ep_norm < 1e-6:
                break

        # If we didn't break, then warn
        else:
            print("WARNING: maximum iterations reached before convergence.")

        return solution, ep, er
