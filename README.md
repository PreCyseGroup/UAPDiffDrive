# Universal Perturbation Attacks on Neural Network Controllers for Wheeled Mobile Robots

**Authors:** Suryaprakash Rajkumar, Ehsan Eslami, Walter Lucia, and Amr Youssef

**Code maintainer:** Suryaprakash Rajkumar

## Install

Use Python 3.10 or newer. Run commands from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

NumPy 2 or newer is required (`np.trapezoid`). CPU is supported; training can use
`--device cuda` when the installed PyTorch and host support it. ROS 2 is optional
and is needed only for the robot node.

## Quick start: test the supplied ROS model

```bash
python scripts/evaluate_policy.py
python scripts/evaluate_policy.py --uap --output-dir outputs/nominal_uap
python scripts/evaluate_policy.py --uap \
  --checkpoint models/sim2real_adversarial_policy.pt \
  --norm models/sim2real_adversarial_norm_stats.npz \
  --output-dir outputs/adversarial_uap
python -m pytest -q
```

The evaluator simulates a held-out lemniscate and writes trajectory NPZ files and
metrics JSON. `--uap` first optimizes one fixed vector on a different reference,
then applies it throughout the test rollout. For a quick execution check add
`--attack-steps 1 --attack-horizon 12`; this is not a research-quality attack.

## Train and simulate

```bash
# Expert collection, behavior cloning, simulation, UAP and Gaussian comparison
python scripts/run_workflow.py --config configs/default.json

# Domain-randomized collection, nominal training, multiscale FGSM defense
python scripts/train_sim2real_pair.py --device auto

# Re-evaluate a newly trained nominal policy
python scripts/evaluate_policy.py \
  --checkpoint artifacts/sim2real_nominal_policy.pt \
  --norm artifacts/sim2real_nominal_norm_stats.npz

# Evaluate the newly trained pair on identical randomized simulation trials
python scripts/evaluate_sim2real_interpolated_wall.py --trials 24
```

Training writes new files under `data/`, `artifacts/`, and `results/`; these are
gitignored. Supplied deployment files stay under `models/`.

## Contents and documentation

| Path | Purpose |
|---|---|
| `src/uap_il/` | Expert, dataset, residual network, training, UAP, kinematics, plots |
| `scripts/` | Training, testing, simulation and dataset tools |
| `models/` | Latest ROS nominal/adversarial checkpoints and paired normalization |
| `configs/default.json` | Portable auto-device settings, epsilon 0.25 |
| `configs/research.json` | Original research configuration snapshot |
| `tests/` | Robot, feature, dataset split, network and attack checks |
| `ros2_nn_trajectory_tracker/` | ROS 2 inference node, launch and configuration |

- [Training and dataset contract](docs/training.md)
- [Testing, simulation and attacks](docs/testing_and_uap.md)
- [Sim-to-real and ROS instructions](docs/sim2real.md)
- [Model identity](models/README.md)
