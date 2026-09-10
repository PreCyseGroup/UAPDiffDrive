# Training and data contract

The expert is the implemented De Luca feedback-linearization controller
(`src/uap_il/controllers.py`). Wheel speeds are clipped before kinematic execution.
The robot uses radius 0.04445 m, wheelbase 0.393 m, wheel limit 10 rad/s and
60 Hz control in the supplied configuration.

The external feature order is `[ex, ey, theta, xdr, ydr, xddr, yddr, vr]`.
Errors are expressed in the robot frame, theta is robot heading on the reference
heading branch, derivatives are reference Cartesian velocity/acceleration, and
vr is reference linear speed. Targets are `[right_wheel, left_wheel]` in rad/s.
Position uses metres, heading radians, velocity m/s, acceleration m/s².

`dataset.py` records inputs, targets, run IDs, states, references, noise flags and
JSON metadata in compressed NumPy archives. Trajectory families include
lemniscate, interpolated, rounded box, circle, ellipse, slalom, sine lane,
Lissajous and trigonometric combinations. Collection size is trajectory based;
sample count varies with trajectory duration and timestep.

DART-style collection adds bounded wheel-action disturbances on a subset of
expert rollouts while preserving clean expert correction labels. State-noise
augmentation and physical/timing randomization are separate controls. This is
expert behavior cloning; aggregation alone does not make it DAgger.

`split_and_normalize(..., run_ids=...)` separates complete rollout IDs and fits
normalization on the training split. At least two rollouts are needed. Passing
no run IDs instead uses a row split and should not be used for paper evaluation.

The network keeps the eight normalized input channels and **appends** raw-heading
cosine and sine, producing ten internal inputs. It is a residual MLP with
512/256/128 widths, GroupNorm, SiLU and dropout, followed by two wheel outputs.
It is not the older five-layer compact MLP. `train.py` uses normalized MSE and
AdamW, validation checkpoint selection and patience-based early stopping.

The sim-to-real trainer materializes multi-scale FGSM samples at normalized
epsilons 0.10/0.20/0.35/0.50 and performs online adversarial/random-input training.
Run `python scripts/train_adversarial_defense.py --help` for individual controls.
Keep each checkpoint with its own normalization file, especially for fine-tuning.

Optional dataset tools:

```bash
# Requires both data/demonstrations.npz and the randomized dataset
python scripts/build_combined_sim2real_dataset.py
python scripts/export_sim2real_nominal_csv.py
```

These utilities preserve aggregation provenance and remap run IDs to avoid
overlapping identifiers.
