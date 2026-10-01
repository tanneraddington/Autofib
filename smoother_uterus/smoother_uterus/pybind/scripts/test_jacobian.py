import numpy as np
from scipy.spatial.transform import Rotation
import ves_solver


if __name__ == "__main__":
    solver = ves_solver.ves_solver.VesSingleArmSolver()

    q = np.array([1, 2, 0.03, 0.015])
    f = np.array([0.5, 0, 0.0])

    solution = solver.forward_kinematics(q, f)
    sol = solution.left_arm[0]
    
    T0 = sol.tip_pose.mean
    p0 = T0[:3,3]
    R0 = T0[:3,:3]

    jac_fd = np.zeros((6,4))
    delta = 1.0e-6

    for i in range(4):
        jv = q.copy()
        jv[i] += delta
        solution = solver.forward_kinematics(jv, f)
        T = solution.left_arm[0].tip_pose.mean
        
        # Spatial frame velocity is easy
        p = T[:3,3]
        jac_fd[3:,i] = (p - p0) / delta
        
        # Spatial frame angular velocity is a bit harder
        R = T[:3,:3]
        R_rel = R @ R0.T
        rotvec = Rotation.from_matrix(R_rel).as_rotvec()
        jac_fd[:3,i] = rotvec / delta

    diff = sol.jac_tip_pose - jac_fd
    np.set_printoptions(precision=6, suppress=True)
    print("Analytical:\n", sol.jac_tip_pose)
    print("Finite differences:\n", jac_fd)
    print("Absolute difference:\n", diff)
    print("Max abs error:", np.max(np.abs(diff)))
        
    