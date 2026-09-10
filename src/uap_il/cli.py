from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uap-il")

import numpy as np

from .attack import learn_closed_loop_uap, learn_uap
from .config import Paths
from .dataset import POLICY_INPUT_DIM, generate_demonstrations, load_dataset, sample_initial_position_offset, split_and_normalize
from .device import resolve_device
from .plotting import (
    save_attack_noise_comparison_summary,
    save_attack_summary,
    save_attack_perturbation_reports,
    save_attack_rollout_perturbation_reports,
    save_gaussian_noise_report,
    save_animation,
    save_dataset_plot,
    save_json,
    save_rollout_plots,
    save_trajectory_family_overview,
    save_training_plot,
)
from .settings import load_settings
from .simulate import compute_metrics, rollout_policy
from .train import load_policy, train_policy
from .trajectories import TrajectorySpec, generate_trajectory, reference_is_feasible, sample_trajectory_specs


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _attack_degradation_score(result: dict) -> float:
    rollout_deltas = [
        rollout["attacked"]["rmse_pos"] - rollout["clean"]["rmse_pos"]
        for rollout in result.get("per_rollout", [])
    ]
    if rollout_deltas:
        return float(max(rollout_deltas))
    return float(result["attacked"]["rmse_pos"] - result["clean"]["rmse_pos"])


def _best_harmful_attack(attack_results: list[dict]) -> dict:
    harmful = [result for result in attack_results if _attack_degradation_score(result) > 0.0]
    return max(harmful, key=_attack_degradation_score) if harmful else {}


def _worst_degraded_rollout_index(clean_metrics: list[dict[str, float]], attacked_metrics: list[dict[str, float]]) -> int:
    increases = [attacked["rmse_pos"] - clean["rmse_pos"] for clean, attacked in zip(clean_metrics, attacked_metrics)]
    return int(np.argmax(increases))


def _close_initial_state(trajectory, rng: np.random.Generator, xy_error: float, theta_error: float) -> np.ndarray:
    if xy_error <= 0.0:
        xy_offset = np.zeros(2, dtype=np.float64)
    else:
        xy_offset = sample_initial_position_offset(rng, min_distance=min(0.10, xy_error), max_distance=xy_error)
    return np.array(
        [
            trajectory.x[0] + xy_offset[0],
            trajectory.y[0] + xy_offset[1],
            trajectory.theta[0] + rng.uniform(-theta_error, theta_error),
        ],
        dtype=np.float64,
    )


def _safe_name(strategy: str, epsilon: float) -> str:
    return f"{strategy}_eps_{epsilon:.3f}".replace(".", "p")


def _validate_feature_shapes(
    data: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    attack_strategies: dict[str, list[float]],
) -> None:
    input_dim = int(data["inputs"].shape[1])
    if input_dim != POLICY_INPUT_DIM:
        raise ValueError(
            f"Dataset has {input_dim} input features, but the current policy expects {POLICY_INPUT_DIM}. "
            "Regenerate the dataset without --skip-data after changing the feature vector."
        )
    stats_dim = int(stats["x_mean"].shape[0])
    if stats_dim != input_dim:
        raise ValueError(
            f"Dataset has {input_dim} input features, but norm stats have {stats_dim}. "
            "Regenerate the dataset and retrain the policy instead of mixing stale artifacts."
        )
    for strategy, mask in attack_strategies.items():
        if len(mask) != input_dim:
            raise ValueError(
                f"Attack mask '{strategy}' has {len(mask)} entries, expected {input_dim}. "
                "Update configs/default.json to match the policy input dimension."
            )


def _primary_attack_strategies(attack_strategies: dict[str, list[float]]) -> dict[str, list[float]]:
    strategy = "ex_ey_theta" if "ex_ey_theta" in attack_strategies else "xy_only"
    if strategy not in attack_strategies:
        raise ValueError("Attack evaluation needs an ex_ey_theta mask in configs/default.json.")
    ignored = sorted(name for name in attack_strategies if name != strategy)
    if ignored:
        print(f"[attack] Ignoring non-primary attack strategies: {', '.join(ignored)}", flush=True)
    return {strategy: attack_strategies[strategy]}


def _load_existing_primary_uap_delta(artifact_dir: Path) -> np.ndarray | None:
    files = sorted(artifact_dir.glob("uap_ex_ey_theta_eps_*.npy"))
    if not files:
        files = sorted(artifact_dir.glob("uap_xy_only_eps_*.npy"))
    files = [path for path in files if not path.name.endswith("_normalized.npy")]
    if not files:
        return None
    return np.load(files[-1]).astype(np.float32)


def _build_lemniscate_eval_specs(seed: int, robot, dt: float, count: int = 3) -> list[TrajectorySpec]:
    rng = np.random.default_rng(seed)
    specs: list[TrajectorySpec] = []
    attempts = 0
    while len(specs) < count:
        attempts += 1
        if attempts > 100 * max(1, count):
            raise RuntimeError("Could not build enough feasible lemniscate evaluation trajectories.")
        spec = TrajectorySpec(
            kind="lemniscate",
            duration=0.0,
            params={
                "eta": float(rng.uniform(0.75, 1.20)),
                "alpha": float(rng.uniform(4.2, 6.0)),
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                "x_offset": 0.0,
                "y_offset": 0.0,
            },
        )
        trajectory = generate_trajectory(dt, spec)
        if reference_is_feasible(trajectory, robot):
            specs.append(spec)
    return specs


def _build_showcase_eval_specs(seed: int, robot, dt: float) -> list[TrajectorySpec]:
    return _build_lemniscate_eval_specs(seed, robot, dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UAP-IL workflow for a differential-drive robot.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--num-runs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--attack-epsilon", type=float)
    parser.add_argument("--attack-method", choices=["closed_loop", "closed_loop_gradient", "output_shift"])
    parser.add_argument("--attack-steps", type=int)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-attack", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(
        args.config,
        {
            "num_runs": args.num_runs,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": args.device,
            "attack_epsilon": args.attack_epsilon,
            "attack_epsilons": [args.attack_epsilon] if args.attack_epsilon is not None else None,
            "attack_method": args.attack_method,
            "attack_steps": args.attack_steps,
        },
    )
    paths = Paths()
    paths.ensure()
    device = resolve_device(settings.device)
    print(f"[device] Using {device}")
    attack_strategies = _primary_attack_strategies(settings.attack_strategies)

    dataset_path = paths.data_dir / "demonstrations.npz"
    checkpoint_path = paths.artifact_dir / "il_policy.pt"
    norm_path = paths.artifact_dir / "norm_stats.npz"
    uap_path = paths.artifact_dir / "uap_delta.npy"
    metrics_path = paths.artifact_dir / "metrics.json"
    attack_analysis_path = paths.artifact_dir / "attack_analysis.json"
    attack_perturbation_csv = paths.artifact_dir / "attack_perturbations.csv"
    attack_perturbation_md = paths.artifact_dir / "attack_perturbations.md"
    attack_rollout_perturbation_csv = paths.artifact_dir / "attack_rollout_perturbations.csv"
    attack_rollout_perturbation_md = paths.artifact_dir / "attack_rollout_perturbations.md"
    gaussian_noise_csv = paths.artifact_dir / "gaussian_noise_baseline.csv"
    gaussian_noise_md = paths.artifact_dir / "gaussian_noise_baseline.md"

    if not args.skip_data or not dataset_path.exists():
        print(f"[data] Generating {settings.num_runs} De Luca expert runs -> {dataset_path}")
        data = generate_demonstrations(
            dataset_path,
            settings.num_runs,
            settings.seed,
            settings.robot,
            settings.simulation,
            settings.deluca_gains,
            state_noise_xy_std=settings.dataset_state_noise.get("xy_std", 0.0),
            state_noise_theta_std=settings.dataset_state_noise.get("theta_std", 0.0),
            state_noise_clip_sigma=settings.dataset_state_noise.get("clip_sigma", 3.0),
            noisy_sample_fraction=settings.dataset_state_noise.get("noisy_sample_fraction", 0.0),
            dart_disturbed_rollout_fraction=settings.dataset_state_noise.get("dart_disturbed_rollout_fraction", 0.0),
            dart_wheel_noise_std=settings.dataset_state_noise.get("dart_wheel_noise_std", 0.0),
            dart_action_clip_sigma=settings.dataset_state_noise.get("dart_action_clip_sigma", 2.0),
            initial_xy_min_distance=settings.dataset_state_noise.get("initial_xy_min_distance", 0.05),
            initial_xy_max_distance=settings.dataset_state_noise.get("initial_xy_max_distance", 0.40),
            initial_theta_error=settings.dataset_state_noise.get("initial_theta_error", 0.25),
            log_progress=True,
        )
    else:
        print(f"[data] Loading existing dataset -> {dataset_path}")
        data = load_dataset(dataset_path)
    if int(data["inputs"].shape[1]) != POLICY_INPUT_DIM:
        raise ValueError(
            f"Loaded dataset has {data['inputs'].shape[1]} input features, but the current policy expects "
            f"{POLICY_INPUT_DIM}. Regenerate data without --skip-data."
        )
    save_dataset_plot(data["inputs"], data["targets"], paths.plot_dir / "dataset_summary.png")
    print(f"[plots] Dataset summary -> {paths.plot_dir / 'dataset_summary.png'}", flush=True)
    overview_specs = sample_trajectory_specs(settings.num_runs, settings.seed, settings.robot, settings.simulation.dt)
    overview_trajs = [generate_trajectory(settings.simulation.dt, spec) for spec in overview_specs]
    save_trajectory_family_overview(overview_trajs, paths.plot_dir / "training_trajectory_families.png")
    print(f"[plots] Trajectory family overview -> {paths.plot_dir / 'training_trajectory_families.png'}", flush=True)

    if not args.skip_train or not checkpoint_path.exists() or not norm_path.exists():
        print(f"[train] Training residual trajectory MLP -> {checkpoint_path}")
        stats = split_and_normalize(data["inputs"], data["targets"], settings.seed, run_ids=data.get("run_ids"))
        history = train_policy(
            data["inputs"],
            data["targets"],
            stats,
            checkpoint_path,
            norm_path,
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            lr=settings.learning_rate,
            seed=settings.seed,
            device=device,
            max_train_samples=settings.max_train_samples,
            max_val_samples=settings.max_val_samples,
            log_progress=True,
        )
        save_training_plot(history, paths.plot_dir / "training_curve.png")
        print(f"[plots] Training curve -> {paths.plot_dir / 'training_curve.png'}", flush=True)
    else:
        print(f"[train] Loading existing policy -> {checkpoint_path}")

    model, stats = load_policy(checkpoint_path, norm_path, device=device)
    _validate_feature_shapes(data, stats, attack_strategies)
    print("[eval] Building fixed lemniscate evaluation trajectories", flush=True)

    eval_specs = _build_showcase_eval_specs(settings.seed + 500, settings.robot, settings.simulation.dt)
    eval_trajs = [generate_trajectory(settings.simulation.dt, spec) for spec in eval_specs]
    eval_rng = np.random.default_rng(settings.seed + 900)
    xy_error = settings.evaluation.get("initial_xy_error", 0.02)
    theta_error = settings.evaluation.get("initial_theta_error", 0.05)
    initial_states = [_close_initial_state(traj, eval_rng, xy_error, theta_error) for traj in eval_trajs]

    representative_clean = rollout_policy(
        model, stats, eval_trajs[0], initial_states[0], settings.robot, settings.simulation
    )
    clean_rollouts = []
    for rollout_id, (traj, init) in enumerate(zip(eval_trajs, initial_states)):
        print(f"[eval] clean rollout {rollout_id + 1}/{len(eval_trajs)} kind={traj.spec.kind}", flush=True)
        clean_rollouts.append(rollout_policy(model, stats, traj, init, settings.robot, settings.simulation))
    clean_metrics = [compute_metrics(rollout) for rollout in clean_rollouts]

    attack_results = []
    gaussian_results = []
    representative_attacked = None
    representative_clean_for_plot = representative_clean
    representative_label = ""
    representative_delta = None
    if not args.skip_attack:
        print(f"[attack] Optimizing UAP sweep using {settings.attack_method}")
        for strategy, mask_values in attack_strategies.items():
            mask = np.asarray(mask_values, dtype=np.float32)
            for epsilon in settings.attack_epsilons:
                print(f"[attack] strategy={strategy} epsilon={epsilon:.3f}", flush=True)
                if settings.attack_method in {"closed_loop", "closed_loop_gradient"}:
                    delta, delta_normalized, attack_metadata = learn_closed_loop_uap(
                        model,
                        stats,
                        eval_trajs,
                        initial_states,
                        settings.robot,
                        settings.simulation,
                        epsilon=epsilon,
                        steps=settings.attack_steps,
                        population_size=settings.attack_population_size,
                        seed=settings.seed + 101,
                        mask=mask,
                        horizon_steps=settings.attack_horizon_steps,
                        score_mode=settings.attack_score_mode,
                        log_progress=True,
                        optimizer="gradient" if settings.attack_method == "closed_loop_gradient" else "blackbox",
                        lr=max(epsilon / 8.0, 1e-4),
                        device=device,
                    )
                elif settings.attack_method == "output_shift":
                    delta, delta_normalized = learn_uap(
                        model,
                        data["inputs"],
                        stats,
                        settings.robot,
                        epsilon=epsilon,
                        steps=settings.attack_steps,
                        batch_size=settings.attack_batch_size,
                        lr=max(epsilon / 8.0, 1e-4),
                        seed=settings.seed + 101,
                        mask=mask,
                        device=device,
                        log_progress=True,
                    )
                    attack_metadata = {
                        "attack_method": "output_shift",
                        "score_mode": "normalized_policy_output_shift",
                    }
                else:
                    raise ValueError(f"Unsupported attack_method: {settings.attack_method}")
                name = _safe_name(strategy, epsilon)
                np.save(paths.artifact_dir / f"uap_{name}.npy", delta)
                np.save(paths.artifact_dir / f"uap_{name}_normalized.npy", delta_normalized)
                attacked_rollouts = []
                for rollout_id, (traj, init) in enumerate(zip(eval_trajs, initial_states)):
                    print(
                        f"[attack] eval rollout {rollout_id + 1}/{len(eval_trajs)} "
                        f"kind={traj.spec.kind} strategy={strategy} eps={epsilon:.3f}",
                        flush=True,
                    )
                    attacked_rollouts.append(
                        rollout_policy(
                            model,
                            stats,
                            traj,
                            init,
                            settings.robot,
                            settings.simulation,
                            perturbation=delta,
                        )
                    )
                attacked_metrics = [compute_metrics(rollout) for rollout in attacked_rollouts]
                per_rollout = [
                    {
                        "rollout_id": rollout_id,
                        "trajectory_kind": eval_trajs[rollout_id].spec.kind,
                        "delta": delta.tolist(),
                        "delta_normalized": delta_normalized.tolist(),
                        "clean": clean_metrics[rollout_id],
                        "attacked": attacked_metrics[rollout_id],
                    }
                    for rollout_id in range(len(eval_trajs))
                ]
                worst_rollout_idx = _worst_degraded_rollout_index(clean_metrics, attacked_metrics)
                result = {
                    "strategy": strategy,
                    "epsilon": float(epsilon),
                    "mask": mask.tolist(),
                    "attack_method": settings.attack_method,
                    "attack_metadata": attack_metadata,
                    "feature_std": stats["x_std"].tolist(),
                    "delta": delta.tolist(),
                    "delta_normalized": delta_normalized.tolist(),
                    "clean": _mean_metrics(clean_metrics),
                    "attacked": _mean_metrics(attacked_metrics),
                    "mean_rmse_pos_increase": float(
                        np.mean(
                            [
                                attacked["rmse_pos"] - clean["rmse_pos"]
                                for clean, attacked in zip(clean_metrics, attacked_metrics)
                            ]
                        )
                    ),
                    "worst_rollout_rmse_pos_increase": float(
                        max(
                            attacked["rmse_pos"] - clean["rmse_pos"]
                            for clean, attacked in zip(clean_metrics, attacked_metrics)
                        )
                    ),
                    "per_rollout": per_rollout,
                    "per_rollout_attacked": attacked_metrics,
                }
                attack_results.append(result)
                save_rollout_plots(
                    clean_rollouts[worst_rollout_idx],
                    attacked_rollouts[worst_rollout_idx],
                    paths.plot_dir,
                    prefix=f"attack_{name}",
                    attacked_label=f"{strategy} eps={epsilon}",
                    delta=delta,
                )
                degradation_score = _attack_degradation_score(result)
                if degradation_score > 0.0 and (
                    representative_attacked is None or degradation_score > representative_attacked[0]
                ):
                    representative_attacked = (degradation_score, attacked_rollouts[worst_rollout_idx])
                    representative_clean_for_plot = clean_rollouts[worst_rollout_idx]
                    representative_label = f"{strategy} eps={epsilon} rollout={worst_rollout_idx}"
                    representative_delta = delta

        print("[eval] Evaluating Gaussian random-noise baseline")
        for strategy, mask_values in attack_strategies.items():
            mask = np.asarray(mask_values, dtype=np.float32)
            for std_normalized in settings.gaussian_noise_stds:
                print(f"[noise] strategy={strategy} std_norm={std_normalized:.3f}", flush=True)
                noisy_rollouts = []
                noisy_metrics = []
                for repeat in range(settings.gaussian_noise_repeats):
                    print(
                        f"[noise] repeat {repeat + 1}/{settings.gaussian_noise_repeats} "
                        f"strategy={strategy} std_norm={std_normalized:.3f}",
                        flush=True,
                    )
                    for rollout_id, (traj, init) in enumerate(zip(eval_trajs, initial_states)):
                        noisy_rollout = rollout_policy(
                            model,
                            stats,
                            traj,
                            init,
                            settings.robot,
                            settings.simulation,
                            gaussian_noise_std_normalized=std_normalized,
                            gaussian_noise_mask=mask,
                            noise_seed=settings.seed + 5000 + repeat * 100 + rollout_id,
                        )
                        noisy_rollouts.append((repeat, rollout_id, noisy_rollout))
                        noisy_metrics.append(compute_metrics(noisy_rollout))
                perturbation_stack = np.vstack([rollout["perturbations"] for _, _, rollout in noisy_rollouts])
                active = mask.astype(bool)
                active_perturbations = perturbation_stack[:, active] if np.any(active) else perturbation_stack
                result = {
                    "strategy": strategy,
                    "std_normalized": float(std_normalized),
                    "mask": mask.tolist(),
                    "repeats": settings.gaussian_noise_repeats,
                    "clean": _mean_metrics(clean_metrics),
                    "attacked": _mean_metrics(noisy_metrics),
                    "mean_abs_raw_noise": float(np.mean(np.abs(active_perturbations))),
                    "rms_raw_noise": float(np.sqrt(np.mean(active_perturbations**2))),
                    "per_rollout_attacked": noisy_metrics,
                }
                gaussian_results.append(result)
                name = f"gaussian_{strategy}_std_{std_normalized:.3f}".replace(".", "p")
                first_noisy = noisy_rollouts[0][2]
                save_rollout_plots(
                    representative_clean,
                    first_noisy,
                    paths.plot_dir,
                    prefix=name,
                    attacked_label=f"Gaussian {strategy} std={std_normalized}",
                    annotation=f"Gaussian noise baseline: {strategy}, normalized std={std_normalized}",
                )

        save_attack_summary(attack_results, paths.plot_dir / "attack_summary.png")
        save_attack_noise_comparison_summary(
            attack_results,
            gaussian_results,
            paths.plot_dir / "attack_vs_gaussian_summary.png",
        )
        save_json({"results": attack_results, "gaussian_results": gaussian_results}, attack_analysis_path)
        save_attack_perturbation_reports(attack_results, attack_perturbation_csv, attack_perturbation_md)
        save_attack_rollout_perturbation_reports(
            attack_results,
            attack_rollout_perturbation_csv,
            attack_rollout_perturbation_md,
        )
        save_gaussian_noise_report(gaussian_results, gaussian_noise_csv, gaussian_noise_md)
        if representative_delta is not None:
            np.save(uap_path, representative_delta)
    elif not attack_analysis_path.exists():
        print("[attack] Looking for an existing primary UAP artifact")
        delta = _load_existing_primary_uap_delta(paths.artifact_dir)
        if delta is not None and delta.shape == representative_clean["features"][0].shape:
            representative_attacked = (
                0.0,
                rollout_policy(
                    model,
                    stats,
                    eval_trajs[0],
                    initial_states[0],
                    settings.robot,
                    settings.simulation,
                    perturbation=delta,
                ),
            )
            representative_label = "loaded UAP"
            representative_delta = delta
        else:
            print("[attack] No compatible primary UAP artifact found")
    else:
        print("[attack] Skipping attack evaluation")

    attacked_for_plot = representative_attacked[1] if representative_attacked else representative_clean_for_plot
    save_rollout_plots(
        representative_clean_for_plot,
        attacked_for_plot,
        paths.plot_dir,
        prefix="rollout",
        attacked_label=representative_label,
        delta=representative_delta,
    )
    save_animation(representative_clean_for_plot, attacked_for_plot, paths.plot_dir / "rollout_animation.gif")
    best_attack = _best_harmful_attack(attack_results)
    if best_attack:
        best_delta = np.asarray(best_attack["delta"], dtype=np.float32)
        for rollout_id, (traj, init, clean_rollout) in enumerate(zip(eval_trajs, initial_states, clean_rollouts)):
            attacked_rollout = rollout_policy(
                model,
                stats,
                traj,
                init,
                settings.robot,
                settings.simulation,
                perturbation=best_delta,
            )
            save_rollout_plots(
                clean_rollout,
                attacked_rollout,
                paths.plot_dir,
                prefix=f"showcase_lemniscate_{rollout_id}",
                attacked_label=f"best {best_attack['strategy']} eps={best_attack['epsilon']}",
                delta=best_delta,
            )
    metrics = {
        "clean_mean": _mean_metrics(clean_metrics),
        "clean_per_rollout": clean_metrics,
        "best_attack": best_attack,
        "gaussian_noise": gaussian_results,
        "eval_trajectory_kinds": [traj.spec.kind for traj in eval_trajs],
        "num_eval_trajectories": len(eval_trajs),
        "wheel_speed_limit_rad_s": settings.robot.wheel_speed_limit,
    }
    save_json(metrics, metrics_path)
    print(f"[eval] Metrics -> {metrics_path}")
    print(f"[eval] Attack analysis -> {attack_analysis_path}")
    print(f"[eval] UAP perturbation report -> {attack_perturbation_md}")
    print(f"[eval] Per-rollout UAP report -> {attack_rollout_perturbation_md}")
    print(f"[eval] Gaussian noise report -> {gaussian_noise_md}")
    print(f"[plots] Figures and animation -> {paths.plot_dir}")
