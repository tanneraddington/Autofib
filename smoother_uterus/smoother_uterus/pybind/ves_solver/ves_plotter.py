import time

import numpy as np
from pathlib import Path
import shutil

import pyvista as pv
import vtk


def remove_close_points(points, eps=1e-6):
    filtered = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - filtered[-1]) > eps:
            filtered.append(p)

    return np.asarray(filtered)


def get_tube_points(points, radius, capping=True):
    spline = pv.Spline(remove_close_points(points), n_points=100)
    tube = spline.tube(radius=radius, capping=capping)

    return tube


def get_tube_poses(poses, radius, capping=True):
    points = np.array([T[:3, 3] for T in poses])
    return get_tube_points(points, radius, capping)


def get_backbone_meshes(sol):
    outer_poses = np.array([T.mean for T in sol.outer_tube_poses])
    inner_poses = np.array([T.mean for T in sol.inner_tube_poses])

    outer_backbone = get_tube_poses(outer_poses, radius=0.001, capping=False)

    R = outer_poses[-1][:3, :3]
    p = outer_poses[-1][:3, 3]

    points = (R.T @ (inner_poses[:, :3, 3] - p).T).T
    mask = points[:, 2] > 1e-4
    inner_after = inner_poses[mask]
    inner_tube_poses = np.vstack([outer_poses[-1][None, :, :], inner_after])

    inner_backbone = get_tube_poses(inner_tube_poses, radius=0.0009)

    return outer_backbone, inner_backbone


def get_ellipsoid_transform(center, cov, scale=1.0, num_sigma=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    one_sigma = np.sqrt(np.maximum(eigvals, 1e-12)) * scale
    radii = num_sigma * one_sigma

    A = eigvecs @ np.diag(radii)
    T = np.eye(4)
    T[:3, :3] = A
    T[:3,  3] = center

    return T
    

def get_arrow(
        start=np.zeros(3), 
        delta=np.ones(3), 
        shaft_radius=0.00015):

    length = np.linalg.norm(delta)

    arrow = pv.Arrow(
        start=start,
        direction=delta,
        scale=length,
        shaft_radius=shaft_radius / length,
        tip_radius=2 * shaft_radius / length,
        tip_length=2 * shaft_radius / length
    )

    return arrow


def get_axes(length=0.003):
    eye = length * np.eye(3)
    return [
        get_arrow(delta=eye[:,0]),
        get_arrow(delta=eye[:,1]),
        get_arrow(delta=eye[:,2])
    ]


def update_actor_transform(actor, matrix):
    transform = actor.GetUserTransform()
    transform.SetMatrix(matrix.flatten().tolist())
    actor.SetVisibility(True)


def init_actor(actor):
    transform = vtk.vtkTransform()
    actor.SetUserTransform(transform)
    actor.SetVisibility(False)


class AnimationFrame:
    def __init__(self, 
                 solution, 
                 p_meas=None, 
                 f_meas=None, 
                 goal_pose=None, 
                 show_backbone_ellipsoids=True, 
                 show_backbone_axes=False):
        # Data
        self.solution = solution
        self.p_meas = p_meas
        self.f_meas = f_meas
        self.goal_pose = goal_pose

        # Plotting flags
        self.show_backbone_ellipsoids=show_backbone_ellipsoids
        self.show_backbone_axes=show_backbone_axes


class VesRobotPlotter:
    def __init__(self):
        pass

    def init_scene(self, plotter, frame):
        solution = frame.solution.left_arm[0]

        # Endoscope mesh
        length = 1.0
        mesh = pv.Cylinder(center=(0, 0, -length / 2 + 1e-4), direction=(0, 0, 1), radius=0.002, height=length)
        plotter.add_mesh(mesh, color="silver", show_edges=True, line_width=2)

        # Backbone meshes
        outer_backbone_mesh, inner_backbone_mesh = get_backbone_meshes(solution)
        self.outer_backbone_actor = plotter.add_mesh(outer_backbone_mesh, color='cobaltgreen', opacity=0.3)
        self.inner_backbone_actor = plotter.add_mesh(inner_backbone_mesh, color='cobalt', opacity=0.3)
        
        # Backbone ellipsoid meshes
        self.backbone_ellipsoid_actors = []
        num_poses = len(solution.outer_tube_poses) + len(solution.inner_tube_poses)
        for _ in range(num_poses):
            ellipsoid = pv.Sphere(radius=1)
            actor = plotter.add_mesh(ellipsoid, color="deepcadmiumred", lighting=False, opacity=0.2)
            init_actor(actor)
            self.backbone_ellipsoid_actors.append(actor)
        
        # Backbone axes meshes
        self.backbone_axes_actors = []
        axes_colors = ['red', 'green', 'blue']
        for _ in range(num_poses):
            axes_actors = []
            for arrow, color in zip(get_axes(), axes_colors):
                actor = plotter.add_mesh(arrow, color=color)
                init_actor(actor)
                axes_actors.append(actor)
            self.backbone_axes_actors.append(axes_actors)

        # Tip force arrow
        self.force_arrow_actor = plotter.add_mesh(get_arrow(), color='deeppink')
        init_actor(self.force_arrow_actor)
        
        # Tip force ellipsoid
        mesh = pv.Sphere(radius=1)
        self.force_ellipsoid_actor = plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
        init_actor(self.force_ellipsoid_actor)

        # GT force arrow
        self.force_meas_arrow_actor = plotter.add_mesh(get_arrow(), color='green')
        init_actor(self.force_meas_arrow_actor)

        # p_meas sphere mesh
        mesh = pv.Sphere(radius=0.0005)
        self.p_meas_actor = plotter.add_mesh(mesh, color='red', lighting=False)
        init_actor(self.p_meas_actor)
        
        # Goal pose mesh
        self.goal_pose_axes_actors = []
        for arrow, color in zip(get_axes(), axes_colors):
            actor = plotter.add_mesh(arrow, color=color)
            init_actor(actor)
            self.goal_pose_axes_actors.append(actor)

        # Text display metadata for solve
        self.text_actor = plotter.add_text('', font_size=10, font='courier')

        focal_point = np.array([0.0, -0.015, 0.025])

        x = focal_point[0] + 0.07
        y = focal_point[1] + 0.1
        z = focal_point[2] + 0.03

        plotter.camera.position = (x, y, z)

        plotter.camera.up = (0, 1, -0.15)
        plotter.camera.focal_point = focal_point

        plotter.add_light(pv.Light(position=(1.0, 1.0, 0.0), intensity=0.5, light_type='scene light'))
        plotter.add_light(pv.Light(position=(0.0, 1.0, 1.0), intensity=0.2, light_type='scene light'))
        plotter.add_light(pv.Light(position=(-1.0, 0.5, -1.0), intensity=0.2, light_type='scene light'))

        plotter.add_axes()
        plotter.enable_depth_peeling(10)
        plotter.enable_anti_aliasing()
    
    def update_backbone_meshes(self, frame):
        sol = frame.solution.left_arm[0]
        outer_backbone, inner_backbone = get_backbone_meshes(sol)

        outer_mesh = self.outer_backbone_actor.mapper.dataset
        outer_mesh.points = outer_backbone.points

        inner_mesh = self.inner_backbone_actor.mapper.dataset
        inner_mesh.points = inner_backbone.points

    def update_backbone_ellipsoids(self, frame):
        if not frame.show_backbone_ellipsoids:
            return
        
        sol = frame.solution.left_arm[0]

        outer_poses_mean = np.array([T.mean for T in sol.outer_tube_poses])
        inner_poses_mean = np.array([T.mean for T in sol.inner_tube_poses])
        outer_poses_cov = np.array([T.cov for T in sol.outer_tube_poses])
        inner_poses_cov = np.array([T.cov for T in sol.inner_tube_poses])

        poses = np.vstack((outer_poses_mean, inner_poses_mean))
        covs = np.vstack((outer_poses_cov, inner_poses_cov))

        for actor, pose, cov in zip(self.backbone_ellipsoid_actors, poses, covs):
            R = pose[:3, :3]
            p = pose[:3, 3]
            cov = R @ (cov[3:, 3:] @ R.T)  # World frame

            matrix = get_ellipsoid_transform(p, cov)
            update_actor_transform(actor, matrix)

    def update_backbone_axes(self, frame):
        if not frame.show_backbone_axes:
            return

        sol = frame.solution.left_arm[0]

        outer_poses_mean = np.array([T.mean for T in sol.outer_tube_poses])
        inner_poses_mean = np.array([T.mean for T in sol.inner_tube_poses])
        poses = np.vstack((outer_poses_mean, inner_poses_mean))
            
        for axes_actors, pose in zip(self.backbone_axes_actors, poses):
            for actor in axes_actors:
                update_actor_transform(actor, pose)

    def update_tip_forces(self, frame):
        # Update vtkTransforms for each actor
        sol = frame.solution.left_arm[0]
        p = sol.inner_tube_poses[-1].mean[:3,3]
        f = sol.tip_force.mean

        force_scale = 0.005
        actor_mesh = self.force_arrow_actor.mapper.dataset
        actor_mesh.points = get_arrow(p, f * force_scale).points

        vis_norm_thresh = 0.01
        vis = True if np.linalg.norm(f) > vis_norm_thresh else False
        self.force_arrow_actor.SetVisibility(vis)
    
        matrix = get_ellipsoid_transform(p + f * force_scale, sol.tip_force.cov, scale=force_scale)
        update_actor_transform(self.force_ellipsoid_actor, matrix)
        
        if frame.f_meas is not None:
            actor_mesh = self.force_meas_arrow_actor.mapper.dataset
            actor_mesh.points = get_arrow(p, frame.f_meas * force_scale).points
            self.force_meas_arrow_actor.SetVisibility(True)

    def update_p_meas(self, frame):
        if frame.p_meas is None:
            return

        matrix = np.eye(4)
        matrix[:3,3] = frame.p_meas
        update_actor_transform(self.p_meas_actor, matrix)

    def update_goal_pose(self, frame):
        if frame.goal_pose is None:
            return
        
        for actor in self.goal_pose_axes_actors:
            update_actor_transform(actor, frame.goal_pose)

    def update_text(self, frame):
        text = (
            f"error:      {frame.solution.error:6.2e} \n"
            f"optimize: {frame.solution.optimize_time_ms:6.2f} ms\n"
            f"total:    {frame.solution.total_time_ms:6.2f} ms\n"
            f"iter:       {frame.solution.iterations}"
        )

        self.text_actor.set_text('upper_left', text)

    def update_actors(self, frame):
        self.update_backbone_meshes(frame)
        self.update_backbone_ellipsoids(frame)
        self.update_p_meas(frame)
        self.update_tip_forces(frame)
        self.update_backbone_axes(frame)
        self.update_goal_pose(frame)
        self.update_text(frame)

    def run(self,
            user_callback,
            total_time_sec=1e5, 
            frame_rate=24, 
            save_movie_path=None,
            show_plot=True):
        
        max_steps = int(float(frame_rate) * float(total_time_sec))

        # if we don't actually want to show the plot, just run the callback and return
        if not show_plot:
            [user_callback(step) for step in range(max_steps)]
            return

        # Init plotter and open movie if required
        pl = pv.Plotter(window_size=(2000, 2000))
        self.init_scene(pl, user_callback(0)) # dummy initial solution
        if save_movie_path is not None:
            pl.open_movie(save_movie_path)

        # Show plotter, but dont block 
        pl.show(interactive_update=True)

        # Run loop of all steps
        for step in range(max_steps):
            # Update all actors with user callback
            frame = user_callback(step)
            self.update_actors(frame)
            pl.update()

            # Write frame already interally renders
            pl.write_frame() if save_movie_path else pl.render()

        pl.close()