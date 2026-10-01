import numpy as np

import ves_solver


def get_q(t):
    q = np.array([
        np.sin(0.1 * t),
        np.sin(0.11 * t),
        0.03 + 0.01 * np.cos(0.12 * t),
        0.01 + 0.01 * np.cos(0.13 * t)
    ])

    return q


if __name__ == "__main__":
    solver = ves_solver.ves_solver.VesSingleArmSolver()
    plotter = ves_solver.ves_plotter.VesRobotPlotter()

    f = np.zeros(3)
    f_cov = np.eye(3)

    def callback(step):
        t = step / 30.0
        q = get_q(t)
        p = np.array((0, 0, 0.035))
        solution = solver.forward_kinematics(q, f, p_meas=p, f_cov=f_cov)

        return ves_solver.ves_plotter.AnimationFrame(
            solution=solution,
            show_backbone_axes=True
        )

    plotter.run(callback)
    