import numpy as np
from scipy.spatial.transform import Rotation

import ves_solver


def clip_q(q):
    lower = [-np.inf, -np.inf, 0.0, 0.0]
    upper = [np.inf, np.inf, 0.06, 0.04]

    return np.clip(q, lower, upper)


def clip_dq(dq, dt):
    rotation_max = np.deg2rad(180)
    translation_max = 0.04
    
    lower = [-rotation_max, -rotation_max, -translation_max, -translation_max]
    upper = [rotation_max, rotation_max, translation_max, translation_max]

    return np.clip(dq / dt, lower, upper) * dt
    

def get_p_goal(t):
    t0, t1 = 0.0, 5.0
    p0 = np.array([0, -0.01, 0.035])
    p1 = np.array([0.00001, 0.01, 0.02])

    alpha = (t - t0) / (t1 - t0)
    alpha = np.clip(alpha, 0.0, 1.0)

    p_goal = (1 - alpha) * p0 + alpha * p1

    return p_goal


if __name__ == "__main__":
    solver = ves_solver.ves_solver.VesSingleArmSolver()
    plotter = ves_solver.ves_plotter.VesRobotPlotter()

    q = np.array([0, 0, 0.03, 0.01])
    f = np.array([0, 0, 0])

    dt = 1.0 / 30

    def callback(step):
        global q, f

        t = step * dt
        p_goal = get_p_goal(t)

        solution = solver.forward_kinematics(q, f)
        sol = solution.left_arm[0]
        p = sol.tip_pose.mean[:3,3]
        Jp = sol.jac_tip_pose[3:]

        e = p_goal - p
        dq = np.linalg.pinv(Jp) @ e
        q = clip_q(q + clip_dq(dq, dt))
        
        return ves_solver.ves_plotter.AnimationFrame(
            solution=solution,
            p_meas=p_goal
        )
    
    plotter.run(callback, total_time_sec=10)
    
