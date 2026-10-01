# mover_uterus

Implements a convenience action client class, `SmootherMover`, for sending `VisualServo` action goals to the `visual_servoing` action server.

The package also provides several executable tools for sending calibration and test trajectories to the robot. These serve both as practical utilities and as minimal examples demonstrating usage of the action interface.

## Maintainer

* James Ferguson (j.m.ferguson@utah.edu)

## Required Package Dependencies

- `smoother_uterus` - Launched with `visual_servoing` action server by default

## Run Calibration Trajectory

`smoother_uterus` starts in calibration by default and must collect calibration data. 
To run the calibration trajectory:

```bash
ros2 run mover_uterus run_calibration
```

This will run and trajectory and also send a signal to `smoother_uterus` that calibration is complete, so that it can exit calibration mode.
Before each arm it waits for `/mover_uterus/continue` (`std_msgs/Empty`) or the task `calibrate` from the uterine `task_publisher_node`.

`go_home()` sends joint command 0 for both arms on `/robot/state/current_state` (the `motor_node` home).

## Run Example Test Trajectories

The test trajectories serve as development tests for visual servoing, but also show how to easily use the `SmootherMover` class to request robot motions.
Given these exmples, using `SmootherMover` in other projects should be straightforward.

Test trajectories can be run by doing:

```bash
ros2 run mover_uterus run_test_trajectory
```

### Command-line Arguments

| Argument            | Options            | Default                 | Description |
|---------------------|--------------------------|--------------------------|-------------|
| `--mode`            | `single`, `both`         | `single`                 | Controls whether one or both arms are commanded. |
| `--arm`             | `left`, `right`          | `left`                   | Specifies which arm to move when `mode=single`. |
| `--frame`           | `<frame_id>` or `image`  | `smoother_uterus/camera`    | Reference frame for the trajectory. Must have a valid TF path to `smoother_uterus/camera`, or be `image` for pixel-space control. |
| `--stop-on-touch`   | flag (bool)              | `false`                  | If set, execution stops immediately when proprioception reports `is_touching == true`. |

The trajectories in the test are always square paths with constant z depth.
After running, the setpoint and measured trajectories will be plotted (requires matplotlib).