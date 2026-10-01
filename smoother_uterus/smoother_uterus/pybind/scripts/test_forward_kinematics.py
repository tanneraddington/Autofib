import numpy as np

import ves_solver


def get_inputs(t):
    q = np.array([
        np.sin(0.1 * t),
        np.sin(0.11 * t),
        0.03 + 0.01 * np.cos(0.12 * t),
        0.01 + 0.01 * np.cos(0.13 * t)
    ])

    f_mag = 5.0 * (-0.5 * np.cos(0.5 * t) + 0.5)**15

    f_dir = np.sin(np.array([0.2, 0.21, 0.22]) * t)
    norm = np.linalg.norm(f_dir)
    f_dir = f_dir / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    f = f_mag * f_dir

    return q, f


if __name__ == "__main__":
    solver = ves_solver.ves_solver.VesSingleArmSolver()
    plotter = ves_solver.ves_plotter.VesRobotPlotter()

    def callback(step):
        t = step / 30.0
        q, f = get_inputs(t)
        solution = solver.forward_kinematics(q, f)

        return ves_solver.ves_plotter.AnimationFrame(
            solution=solution,
            show_backbone_axes=True
        )

    plotter.run(callback)
    