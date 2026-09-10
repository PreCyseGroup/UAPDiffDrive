# Testing, simulation and UAP

`python -m pytest -q` runs unit tests. `scripts/evaluate_policy.py` executes the
trained model in closed loop and optionally generates a UAP. Clean and attacked
tests share the initial condition and reference. Results include positional
RMSE/max error, integrated positional error, and heading RMSE/max error.

The primary mask is `[1,1,1,0,0,0,0,0]`. The optimized fixed perturbation obeys
`max(abs(delta_normalized)) <= epsilon`. Deployment uses
`delta_raw = delta_normalized * x_std`, then normalizes the perturbed features.
Epsilon 0.25 is a normalized feature bound, not 0.25 metres or radians. The
evaluator saves both raw and normalized vectors. Convert using the target
model's statistics when transferring a normalized perturbation between models.

UAP is constant across steps. FGSM is recalculated from the current input and
gradient; Gaussian noise is stochastic. The code also contains output-shift and
black-box UAP objectives. State the objective, mask, epsilon, optimization
horizon, steps, seed and application window when reporting an experiment.

The new evaluator fits UAP on alpha=4/phase=0 and tests alpha=5/phase=0.7. This
separates attack fitting and test references, but a single reference is a small
demonstration, not comprehensive generalization evidence. Evaluate multiple
independent reference families/seeds for a paper.

The inherited `run_workflow.py` fits and scores its closed-loop attack on the
same evaluation references; its scores measure attack optimization performance,
not transfer to unseen attack trajectories. It also generates plots and GIFs.

`interpolated_wall_robustness_evaluation.py` compares clean/UAP/FGSM conditions.
Provide checkpoint and normalization paths with `--baseline-checkpoint`,
`--baseline-norm`, `--defense-checkpoint`, and `--defense-norm`. It loads raw UAPs
from `artifacts/uap_ex_ey_theta_eps_0p250.npy` (and other requested epsilon tags).
Generate the matching UAP first; missing vectors are reported and skipped.
The filename and scaling must match the baseline policy's normalization.
`interpolated_wall_showcase.py` supplies the wall simulation and figure helpers.

The wall is a geometric collision/clearance benchmark, not a contact dynamics
simulator. Randomized simulation uses the same sampled domains for both policies.
Fresh outputs are generated only when commands run and should remain gitignored.
