import numpy as np

import ves_solver


yc = -0.015
zc = 0.025
r = 0.012

def get_goal_pose(t):
    px = r * np.cos(0.3 * t)
    py = yc
    pz = zc + r * np.sin(0.3 * t)

    pose = np.eye(4)
    pose[:3,3] = [px, py, pz]

    return pose


if __name__ == "__main__":
    
    solver = ves_solver.ves_solver.VesSingleArmSolver()
    plotter = ves_solver.ves_plotter.VesRobotPlotter()
        
    def callback(step):
        t = step / 30.0
        pose = get_goal_pose(t)
        solution, ep, er = solver.inverse_kinematics(pose, print_iter=True)
        
        return ves_solver.ves_plotter.AnimationFrame(
            solution=solution,
            goal_pose=pose,
            show_backbone_ellipsoids=False,
            show_backbone_axes=True
        )
    
    save_movie_path = None
    # save_movie_path = 'asdf.mp4'
    plotter.run(callback, total_time_sec=10, save_movie_path=save_movie_path)
    