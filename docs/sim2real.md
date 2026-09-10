# Sim-to-real and ROS 2

The latest ROS pair is supplied in `models/`: `sim2real_nominal_policy.pt` and
`sim2real_adversarial_policy.pt`, each with its matching `*_norm_stats.npz`.
These are the user-confirmed robot models. No robot recordings are bundled.

Training randomizes 85% of rollouts: radius/base scales 0.94–1.06, independent
wheel gains 0.88–1.12, motor time constant 0.02–0.12 s, action delay 0–3 steps,
observation delay 0–2 steps. State noise and DART recovery collection are also
enabled. These are synthetic training ranges, not measured robot calibration.

`evaluate_sim2real_interpolated_wall.py` evaluates newly trained model files in
`artifacts/` by default. To evaluate the supplied pair, first copy the four
`models/` files to a newly created `artifacts/` directory. Do not overwrite a
training run you want to retain. `--include-legacy` requires the older checkpoints.

## Robot node

In a sourced ROS 2 workspace with colcon, place `ros2_nn_trajectory_tracker/`
inside the workspace's `src/` directory. Its Python must have NumPy and PyTorch.
Build and source the workspace:

```bash
colcon build --packages-select nn_trajectory_tracker
source install/setup.bash
ros2 launch nn_trajectory_tracker lemniscate_tracker.launch.py \
  model_path:=/absolute/path/to/paper_companion/models/sim2real_nominal_policy.pt \
  norm_stats_path:=/absolute/path/to/paper_companion/models/sim2real_nominal_norm_stats.npz \
  max_linear_velocity:=0.2 max_angular_velocity:=0.8
```

Select the adversarial model by changing **both** paths. Paths are explicit
launch arguments; model files need not be copied into the ROS package.
Defaults are `/odemetry/filtered` (spelling retained from the robot interface),
`/cmd_vel`, `vicon/world`, CPU and 60 Hz. Override topic/frame launch arguments
to match the robot. Check odometry, coordinate conventions, wheel ordering,
command scaling and stale-odometry behavior before enabling motion.

The node converts predicted right/left wheel speeds to linear/angular velocity.
It receives eight raw features and appends heading trig features internally.
No inference-time smoothing is added. The package supplies clean policy tracking;
it does not implement the real-time UAP/FGSM experiment harness. Attacks in this
release run in simulation. ROS build, middleware and physical motion require
validation on the target machine.
