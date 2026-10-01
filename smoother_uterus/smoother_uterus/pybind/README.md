# Pybind smoother_uterus Single Arm Model

`smoother_uterus` kinematics/mechanics models you can run from Python.
These are useful for testing kinematics (e.g. for planning suturing), calibrating the model,
and running simulations. The models give access to **uncertainty** of the VES arm,
which is useful for planning under uncertainty.

## Install

1. Install GTSAM, the optimization library used for these models.
   If you're using the GTSAM build from `smoother_uterus`, point to the local build tree:
   ```python
   # In setup.py, top of file:
   gtsam_include = [
       "/path/to/smoother_uterus/build/smoother_uterus/_deps/gtsam-build",
       "/path/to/smoother_uterus/build/smoother_uterus/_deps/gtsam-src",
   ]
   gtsam_lib = ["/path/to/smoother_uterus/build/smoother_uterus/_deps/gtsam-build/gtsam"]
   ```
   The defaults (`/usr/local/include`, `/usr/local/lib`) work for a system-installed GTSAM.

2. Create and source a Python venv:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Build and install the pybind package:
   ```bash
   pip install . -v
   ```

## Running Examples

With the venv active, run any script from the `scripts/` directory:

```bash
python scripts/test_forward_kinematics.py
python scripts/test_inverse_kinematics.py
python scripts/test_jacobian.py
python scripts/test_jacobian_control.py
python scripts/test_tip_position_meas.py
```

## Python API

The main entry point is `ves_solver.VesSingleArmSolver`:

```python
import ves_solver

solver = ves_solver.VesSingleArmSolver(window_size=1)

# Forward kinematics: q = [theta1, theta2, z_outer, z_inner]
q = np.array([0.5, -0.3, 0.03, 0.015])
solution = solver.forward_kinematics(q)

# Access tip pose and Jacobian
tip = solution.left_arm[0].tip_pose.mean   # 4x4 SE(3) matrix
J   = solution.left_arm[0].jac_tip_pose    # 6x4 Jacobian (angular over position)

# With force and optional position measurement
solution = solver.forward_kinematics(q, f=np.array([0.1, 0, 0]))
solution = solver.forward_kinematics(q, p_meas=np.array([0, 0, 0.035]))
```

### C++ module (`smoother_uterus`)

The underlying C++ module is `smoother_uterus`. `SmootherSolver` is bound directly:

```python
import smoother_uterus

solver = smoother_uterus.SmootherSolver()   # uses a dummy camera (pixels unused in Python)
sample = smoother_uterus.SingleArmSample()
sample.time_seconds = 0.0
sample.joint_values = np.array([0.5, -0.3, 0.03, 0.015])
sample.tip_force = smoother_uterus.Vector3Gaussian(np.zeros(3), 1e-6 * np.eye(3))

solution = solver.solve([sample], [])    # (left_samples, right_samples)
```

Available setters on `SmootherSolver`:
- `set_calibration(SmootherCalibration)`
- `set_inner_tube_noise_std(Vector6)`
- `set_outer_tube_noise_std(Vector6)`
- `set_stiffness_params(outer_k_bending, inner_k_bending, k_torsion)`
- `set_tip_accel_prior_std(std)`
- `set_left_tip_offset(Vector3)` / `set_right_tip_offset(Vector3)`
- `set_pixel_meas_std(std)`
