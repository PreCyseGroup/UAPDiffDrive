from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import RobotParams, SimParams
from .dataset import POLICY_INPUT_DIM
from .model import WheelMLP
from .simulate import compute_metrics, rollout_policy
from .trajectories import Trajectory


def _project_linf_masked(delta: torch.Tensor, epsilon: float, mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp(delta, -epsilon, epsilon) * mask


def project_linf_masked_torch(delta: torch.Tensor, epsilon: float, mask: torch.Tensor) -> torch.Tensor:
    """Project a normalized delta into a masked L-infinity ball."""
    return torch.clamp(delta, -epsilon, epsilon) * mask


def _project_linf_masked_np(delta: np.ndarray, epsilon: float, mask: np.ndarray) -> np.ndarray:
    return np.clip(delta, -epsilon, epsilon).astype(np.float32) * mask.astype(np.float32)


def angle_wrap_torch(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] with differentiable torch ops."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def unwrap_heading_to_reference_torch(theta: torch.Tensor, reference_theta: torch.Tensor) -> torch.Tensor:
    """Represent a wrapped robot heading on the same continuous branch as the reference heading."""
    return reference_theta + angle_wrap_torch(theta - reference_theta)


def dd_step_torch(state: torch.Tensor, control: torch.Tensor, dt: float) -> torch.Tensor:
    """Differential-drive/unicycle kinematic step for [x, y, theta]."""
    x, y, theta = state.unbind()
    v, omega = control.unbind()
    return torch.stack(
        [
            x + dt * v * torch.cos(theta),
            y + dt * v * torch.sin(theta),
            angle_wrap_torch(theta + dt * omega),
        ]
    )


def _wheel_to_unicycle_torch(wheels: torch.Tensor, robot_params: RobotParams) -> torch.Tensor:
    wr, wl = wheels.unbind()
    v = robot_params.wheel_radius * (wr + wl) / 2.0
    omega = robot_params.wheel_radius * (wr - wl) / robot_params.wheelbase
    return torch.stack([v, omega])


def _trajectory_tensors(
    trajectory: Trajectory,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "x": torch.as_tensor(trajectory.x, dtype=torch.float32, device=device),
        "y": torch.as_tensor(trajectory.y, dtype=torch.float32, device=device),
        "xd": torch.as_tensor(trajectory.xd, dtype=torch.float32, device=device),
        "yd": torch.as_tensor(trajectory.yd, dtype=torch.float32, device=device),
        "xdd": torch.as_tensor(trajectory.xdd, dtype=torch.float32, device=device),
        "ydd": torch.as_tensor(trajectory.ydd, dtype=torch.float32, device=device),
        "theta": torch.as_tensor(trajectory.theta, dtype=torch.float32, device=device),
        "v": torch.as_tensor(trajectory.v, dtype=torch.float32, device=device),
    }


def build_policy_input_torch(
    state: torch.Tensor,
    trajectory_t: dict[str, torch.Tensor],
    k: int,
    lookahead_steps: int,
) -> torch.Tensor:
    """Torch mirror of dataset.tracking_features."""
    feature_k = min(k + lookahead_steps, int(trajectory_t["x"].numel()) - 1)
    dx = trajectory_t["x"][feature_k] - state[0]
    dy = trajectory_t["y"][feature_k] - state[1]
    cos_theta = torch.cos(state[2])
    sin_theta = torch.sin(state[2])
    ex = cos_theta * dx + sin_theta * dy
    ey = -sin_theta * dx + cos_theta * dy
    theta = unwrap_heading_to_reference_torch(state[2], trajectory_t["theta"][k])
    return torch.stack(
        [
            ex,
            ey,
            theta,
            trajectory_t["xd"][k],
            trajectory_t["yd"][k],
            trajectory_t["xdd"][k],
            trajectory_t["ydd"][k],
            trajectory_t["v"][k],
        ]
    )


def fgsm_input_delta(
    model: WheelMLP,
    stats: dict[str, np.ndarray],
    feature: np.ndarray,
    epsilon: float,
    mask: np.ndarray,
    target_mode: str = "reverse_output",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-step FGSM perturbation for the current policy input.

    The attack is targeted toward a bad wheel-command surrogate because the
    closed-loop tracking loss is not differentiable through the NumPy simulator.
    `reverse_output` pushes the normalized policy output toward the opposite of
    the clean command at the current state.
    """

    device = next(model.parameters()).device
    x_mean = torch.from_numpy(stats["x_mean"].astype(np.float32)).to(device)
    x_std = torch.from_numpy(stats["x_std"].astype(np.float32)).to(device)
    mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
    x_raw = torch.from_numpy(feature.astype(np.float32)).to(device)
    x_norm = ((x_raw - x_mean) / x_std).detach().requires_grad_(True)

    clean_y = model(x_norm.unsqueeze(0))[0].detach()
    if target_mode == "reverse_output":
        target_y = -clean_y
    elif target_mode == "zero_output":
        target_y = torch.zeros_like(clean_y)
    else:
        raise ValueError(f"Unsupported FGSM target_mode: {target_mode}")

    y = model(x_norm.unsqueeze(0))[0]
    loss = F.mse_loss(y, target_y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    if x_norm.grad is None:
        delta_norm = torch.zeros_like(x_norm)
    else:
        # Targeted FGSM: step in the negative gradient direction toward target_y.
        delta_norm = -epsilon * torch.sign(x_norm.grad.detach()) * mask_t
    delta_norm = torch.clamp(delta_norm, -epsilon, epsilon) * mask_t
    delta_norm_np = delta_norm.detach().cpu().numpy().astype(np.float32)
    delta_raw_np = delta_norm_np * stats["x_std"].astype(np.float32)
    return delta_raw_np.astype(np.float32), delta_norm_np


def _truncate_trajectory(trajectory: Trajectory, max_steps: int | None) -> Trajectory:
    if max_steps is None or max_steps <= 0 or max_steps >= len(trajectory.t):
        return trajectory
    count = max(2, int(max_steps))
    return Trajectory(
        t=trajectory.t[:count],
        x=trajectory.x[:count],
        y=trajectory.y[:count],
        xd=trajectory.xd[:count],
        yd=trajectory.yd[:count],
        xdd=trajectory.xdd[:count],
        ydd=trajectory.ydd[:count],
        theta=trajectory.theta[:count],
        v=trajectory.v[:count],
        omega=trajectory.omega[:count],
        spec=trajectory.spec,
    )


def _closed_loop_score(
    model: WheelMLP,
    stats: dict[str, np.ndarray],
    trajectories: list[Trajectory],
    initial_states: list[np.ndarray],
    clean_metrics: list[dict[str, float]],
    robot_params: RobotParams,
    sim_params: SimParams,
    delta_norm: np.ndarray,
    score_mode: str,
) -> tuple[float, list[dict[str, float]]]:
    delta_raw = delta_norm.astype(np.float32) * stats["x_std"].astype(np.float32)
    attacked_metrics = []
    increases = []
    for trajectory, initial_state, clean in zip(trajectories, initial_states, clean_metrics):
        rollout = rollout_policy(
            model,
            stats,
            trajectory,
            initial_state,
            robot_params,
            sim_params,
            perturbation=delta_raw,
        )
        attacked = compute_metrics(rollout)
        attacked_metrics.append(attacked)
        increases.append(attacked["rmse_pos"] - clean["rmse_pos"])

    if score_mode == "worst":
        score = max(increases)
    elif score_mode == "mean":
        score = float(np.mean(increases))
    else:
        raise ValueError(f"Unsupported closed-loop attack score_mode: {score_mode}")
    return float(score), attacked_metrics


def _validate_closed_loop_attack_inputs(
    stats: dict[str, np.ndarray],
    trajectories: list[Trajectory],
    initial_states: list[np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    if not trajectories:
        raise ValueError("Closed-loop UAP needs at least one trajectory.")
    if len(initial_states) != len(trajectories):
        raise ValueError(
            f"Closed-loop UAP got {len(initial_states)} initial states for {len(trajectories)} trajectories."
        )
    input_dim = int(stats["x_std"].shape[0])
    if input_dim != POLICY_INPUT_DIM:
        raise ValueError(f"Closed-loop UAP expected {POLICY_INPUT_DIM} policy inputs, got {input_dim}.")
    mask_np = mask.astype(np.float32)
    if mask_np.shape[0] != input_dim:
        raise ValueError(f"Attack mask has {mask_np.shape[0]} entries, expected {input_dim}.")
    if np.count_nonzero(mask_np > 0.0) == 0:
        raise ValueError("Closed-loop UAP needs at least one active mask entry.")
    return mask_np


def closed_loop_tracking_loss_torch(
    model: WheelMLP,
    stats_t: dict[str, torch.Tensor],
    trajectory_t: dict[str, torch.Tensor],
    initial_state: torch.Tensor,
    robot_params: RobotParams,
    sim_params: SimParams,
    delta: torch.Tensor,
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return attacked position RMSE for one differentiable rollout."""
    total_steps = int(trajectory_t["x"].numel())
    rollout_steps = total_steps if horizon_steps is None else min(total_steps, max(1, int(horizon_steps)))
    lookahead_steps = max(0, int(round(sim_params.lookahead_time / sim_params.dt)))
    state = initial_state
    pos_sq_errors = []
    wheel_limit = float(robot_params.wheel_speed_limit)

    for k in range(rollout_steps):
        dx = trajectory_t["x"][k] - state[0]
        dy = trajectory_t["y"][k] - state[1]
        pos_sq_errors.append(dx * dx + dy * dy)

        z_raw = build_policy_input_torch(state, trajectory_t, k, lookahead_steps)
        z_norm = (z_raw - stats_t["x_mean"]) / stats_t["x_std"]
        z_attacked = z_norm + delta
        u_norm = model(z_attacked.unsqueeze(0)).squeeze(0)
        wheels = u_norm * stats_t["y_std"] + stats_t["y_mean"]
        wheels = torch.clamp(wheels, -wheel_limit, wheel_limit)
        control = _wheel_to_unicycle_torch(wheels, robot_params)
        state = dd_step_torch(state, control, sim_params.dt)

    return torch.sqrt(torch.mean(torch.stack(pos_sq_errors)) + 1e-12)


def _fooling_rate(
    model: WheelMLP,
    x_norm: torch.Tensor,
    clean_y_norm: torch.Tensor,
    delta_norm: torch.Tensor,
    threshold: float,
    batch_size: int,
) -> float:
    fooled = 0
    total = x_norm.shape[0]
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            attacked_y = model(x_norm[start:end] + delta_norm)
            shift = torch.linalg.vector_norm(attacked_y - clean_y_norm[start:end], dim=1)
            fooled += int(torch.count_nonzero(shift >= threshold).item())
    return fooled / max(total, 1)


def _minimal_linearized_output_shift(
    model: WheelMLP,
    x_adv_norm: torch.Tensor,
    clean_y_norm: torch.Tensor,
    threshold: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """DeepFool-style per-sample correction for a regression output boundary.

    The original UAP algorithm uses a per-sample minimal perturbation that crosses
    a classifier decision boundary. For this wheel-speed regressor, the analogous
    boundary is a normalized output displacement of at least ``threshold`` from the
    clean policy output.
    """

    x_var = x_adv_norm.detach().clone().requires_grad_(True)
    y = model(x_var.unsqueeze(0))[0]
    output_delta = y.detach() - clean_y_norm
    current_shift = torch.linalg.vector_norm(output_delta)
    if current_shift >= threshold:
        return torch.zeros_like(x_adv_norm)

    best_step = None
    best_norm = None
    overshoot = 1.02
    for output_idx in range(y.numel()):
        for sign in (-1.0, 1.0):
            model.zero_grad(set_to_none=True)
            if x_var.grad is not None:
                x_var.grad.zero_()
            margin = sign * (y[output_idx] - clean_y_norm[output_idx])
            margin.backward(retain_graph=True)
            grad = x_var.grad.detach() * mask
            grad_norm_sq = torch.dot(grad, grad)
            if float(grad_norm_sq.item()) <= 1e-12:
                continue
            distance = threshold - float((sign * output_delta[output_idx]).item())
            if distance <= 0.0:
                return torch.zeros_like(x_adv_norm)
            step = overshoot * distance * grad / grad_norm_sq
            step_norm = torch.linalg.vector_norm(step)
            if best_norm is None or step_norm < best_norm:
                best_norm = step_norm
                best_step = step.detach()

    if best_step is None:
        return torch.zeros_like(x_adv_norm)
    return best_step * mask


def learn_uap(
    model: WheelMLP,
    inputs: np.ndarray,
    stats: dict[str, np.ndarray],
    robot_params: RobotParams,
    epsilon: float,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    mask: np.ndarray | None = None,
    device: torch.device | str | None = None,
    fooling_threshold: float = 0.5,
    target_fooling_rate: float = 0.8,
    log_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    model.eval()

    max_points = min(batch_size, len(inputs))
    sample_idx = rng.choice(len(inputs), size=max_points, replace=False)
    x_raw_all = torch.from_numpy(inputs[sample_idx].astype(np.float32)).to(device)
    x_mean = torch.from_numpy(stats["x_mean"]).to(device)
    x_std = torch.from_numpy(stats["x_std"]).to(device)
    input_dim = inputs.shape[1]
    delta_norm = torch.zeros(input_dim, dtype=torch.float32, device=device)
    mask_t = (
        torch.ones(input_dim, dtype=torch.float32, device=device)
        if mask is None
        else torch.from_numpy(mask.astype(np.float32)).to(device)
    )
    if mask_t.numel() != input_dim:
        raise ValueError(f"Attack mask has {mask_t.numel()} entries, expected {input_dim}.")

    x_norm_all = (x_raw_all - x_mean) / x_std
    with torch.no_grad():
        clean_y_norm_all = model(x_norm_all)
    if log_progress:
        print(
            f"[attack] UAP samples={max_points} epsilon={epsilon:.3f} "
            f"passes={steps} threshold={fooling_threshold:.3f}",
            flush=True,
        )

    order = np.arange(max_points)
    max_passes = max(1, steps)
    eval_batch_size = min(max_points, 4096)
    for pass_idx in range(max_passes):
        rng.shuffle(order)
        for idx in order:
            x_i = x_norm_all[int(idx)]
            clean_y_i = clean_y_norm_all[int(idx)]
            with torch.no_grad():
                attacked_y_i = model((x_i + delta_norm).unsqueeze(0))[0]
                already_fooled = torch.linalg.vector_norm(attacked_y_i - clean_y_i) >= fooling_threshold
            if bool(already_fooled.item()):
                continue
            correction = _minimal_linearized_output_shift(
                model,
                x_i + delta_norm,
                clean_y_i,
                fooling_threshold,
                mask_t,
            )
            delta_norm = _project_linf_masked(delta_norm + correction, epsilon, mask_t).detach()

        rate = _fooling_rate(model, x_norm_all, clean_y_norm_all, delta_norm, fooling_threshold, eval_batch_size)
        if log_progress:
            print(
                f"[attack] pass {pass_idx + 1:02d}/{max_passes:02d} "
                f"fooling_rate={rate:.3f}",
                flush=True,
            )
        if rate >= target_fooling_rate:
            break

    delta_norm_np = (delta_norm * mask_t).detach().cpu().numpy().astype(np.float32)
    delta_raw_np = delta_norm_np * stats["x_std"]
    return delta_raw_np.astype(np.float32), delta_norm_np


def learn_closed_loop_uap_gradient(
    model: WheelMLP,
    stats: dict[str, np.ndarray],
    trajectories: list[Trajectory],
    initial_states: list[np.ndarray],
    robot_params: RobotParams,
    sim_params: SimParams,
    epsilon: float,
    steps: int,
    lr: float,
    mask: np.ndarray,
    horizon_steps: int | None = None,
    score_mode: str = "mean",
    log_progress: bool = False,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Optimize a shared normalized delta with differentiable closed-loop rollout."""
    mask_np = _validate_closed_loop_attack_inputs(stats, trajectories, initial_states, mask)
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)
    model.eval()

    stats_t = {
        "x_mean": torch.as_tensor(stats["x_mean"].astype(np.float32), dtype=torch.float32, device=device),
        "x_std": torch.as_tensor(stats["x_std"].astype(np.float32), dtype=torch.float32, device=device),
        "y_mean": torch.as_tensor(stats["y_mean"].astype(np.float32), dtype=torch.float32, device=device),
        "y_std": torch.as_tensor(stats["y_std"].astype(np.float32), dtype=torch.float32, device=device),
    }
    mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device)
    input_dim = int(stats_t["x_std"].numel())
    trajectory_ts = [_trajectory_tensors(_truncate_trajectory(traj, horizon_steps), device) for traj in trajectories]
    initial_state_ts = [
        torch.as_tensor(initial_state.astype(np.float32), dtype=torch.float32, device=device)
        for initial_state in initial_states
    ]
    attack_trajs = [_truncate_trajectory(trajectory, horizon_steps) for trajectory in trajectories]

    clean_metrics = [
        compute_metrics(rollout_policy(model, stats, trajectory, initial_state, robot_params, sim_params))
        for trajectory, initial_state in zip(attack_trajs, initial_states)
    ]

    # This is a differentiable surrogate closed-loop UAP: one bounded delta is
    # shared across trajectories and time. It is not the original DeepFool UAP.
    delta = torch.zeros(input_dim, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))
    max_steps = max(1, int(steps))
    final_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    best_loss = torch.tensor(-float("inf"), dtype=torch.float32, device=device)
    best_delta = torch.zeros_like(delta.detach())
    completed_steps = 0
    stopped_early_reason: str | None = None

    for step_idx in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        losses = [
            closed_loop_tracking_loss_torch(
                model,
                stats_t,
                trajectory_t,
                initial_state_t,
                robot_params,
                sim_params,
                delta * mask_t,
                horizon_steps=None,
            )
            for trajectory_t, initial_state_t in zip(trajectory_ts, initial_state_ts)
        ]
        final_loss = torch.mean(torch.stack(losses))
        if not torch.isfinite(final_loss):
            stopped_early_reason = "nonfinite_loss"
            with torch.no_grad():
                delta.copy_(best_delta)
            if log_progress:
                print(
                    f"[attack] closed-loop-gradient step {step_idx + 1:02d}/{max_steps:02d} "
                    "stopping: non-finite attacked_rmse; restoring best finite perturbation",
                    flush=True,
                )
            break
        if final_loss.detach() > best_loss:
            best_loss = final_loss.detach()
            best_delta = project_linf_masked_torch(delta.detach(), epsilon, mask_t)
        (-final_loss).backward()
        if delta.grad is None or not torch.all(torch.isfinite(delta.grad)):
            stopped_early_reason = "nonfinite_gradient"
            with torch.no_grad():
                delta.copy_(best_delta)
            if log_progress:
                print(
                    f"[attack] closed-loop-gradient step {step_idx + 1:02d}/{max_steps:02d} "
                    "stopping: non-finite gradient; restoring best finite perturbation",
                    flush=True,
                )
            break
        torch.nn.utils.clip_grad_norm_([delta], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            delta.copy_(project_linf_masked_torch(delta, epsilon, mask_t))
            if not torch.all(torch.isfinite(delta)):
                stopped_early_reason = "nonfinite_delta"
                delta.copy_(best_delta)
                if log_progress:
                    print(
                        f"[attack] closed-loop-gradient step {step_idx + 1:02d}/{max_steps:02d} "
                        "stopping: non-finite perturbation; restoring best finite perturbation",
                        flush=True,
                    )
                break
        completed_steps = step_idx + 1
        if log_progress:
            print(
                f"[attack] closed-loop-gradient step {step_idx + 1:02d}/{max_steps:02d} "
                f"attacked_rmse={float(final_loss.detach().cpu().item()):.6f}",
                flush=True,
            )

    with torch.no_grad():
        final_losses = [
            closed_loop_tracking_loss_torch(
                model,
                stats_t,
                trajectory_t,
                initial_state_t,
                robot_params,
                sim_params,
                delta * mask_t,
                horizon_steps=None,
            )
            for trajectory_t, initial_state_t in zip(trajectory_ts, initial_state_ts)
        ]
        final_loss = torch.mean(torch.stack(final_losses))
        if not torch.isfinite(final_loss):
            final_loss = best_loss if torch.isfinite(best_loss) else torch.tensor(0.0, dtype=torch.float32, device=device)

    delta_norm_np = project_linf_masked_torch(delta.detach(), epsilon, mask_t).cpu().numpy().astype(np.float32)
    delta_raw_np = delta_norm_np * stats["x_std"].astype(np.float32)
    attacked_metrics = [
        compute_metrics(
            rollout_policy(
                model,
                stats,
                trajectory,
                initial_state,
                robot_params,
                sim_params,
                perturbation=delta_raw_np,
            )
        )
        for trajectory, initial_state in zip(attack_trajs, initial_states)
    ]
    increases = [attacked["rmse_pos"] - clean["rmse_pos"] for clean, attacked in zip(clean_metrics, attacked_metrics)]
    if score_mode == "worst":
        score = max(increases)
    elif score_mode == "mean":
        score = float(np.mean(increases))
    else:
        raise ValueError(f"Unsupported closed-loop attack score_mode: {score_mode}")
    metadata = {
        "attack_method": "closed_loop_gradient",
        "optimizer": "Adam",
        "score_mode": score_mode,
        "score": float(score),
        "mean_rmse_pos_increase": float(np.mean(increases)),
        "worst_rollout_rmse_pos_increase": float(max(increases)),
        "epsilon": float(epsilon),
        "lr": float(lr),
        "steps": int(max_steps),
        "completed_steps": int(completed_steps),
        "stopped_early_reason": stopped_early_reason,
        "horizon_steps": None if horizon_steps is None else int(horizon_steps),
        "final_attacked_rmse": float(final_loss.detach().cpu().item()),
        "clean_metrics": clean_metrics,
        "attacked_metrics": attacked_metrics,
        "optimization_clean_metrics": clean_metrics,
        "optimization_attacked_metrics": attacked_metrics,
    }
    return delta_raw_np.astype(np.float32), delta_norm_np, metadata


def learn_closed_loop_uap_blackbox(
    model: WheelMLP,
    stats: dict[str, np.ndarray],
    trajectories: list[Trajectory],
    initial_states: list[np.ndarray],
    robot_params: RobotParams,
    sim_params: SimParams,
    epsilon: float,
    steps: int,
    population_size: int,
    seed: int,
    mask: np.ndarray,
    horizon_steps: int | None = None,
    score_mode: str = "mean",
    log_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Optimize a shared delta with black-box random/population search.

    This fallback does not use gradients. It directly evaluates candidate
    universal deltas in closed loop and keeps the delta that maximizes position
    RMSE increase.
    """
    mask_np = _validate_closed_loop_attack_inputs(stats, trajectories, initial_states, mask)
    input_dim = int(stats["x_std"].shape[0])
    active_dims = np.flatnonzero(mask_np > 0.0)

    rng = np.random.default_rng(seed)
    attack_trajs = [_truncate_trajectory(trajectory, horizon_steps) for trajectory in trajectories]
    clean_metrics = [
        compute_metrics(rollout_policy(model, stats, trajectory, initial_state, robot_params, sim_params))
        for trajectory, initial_state in zip(attack_trajs, initial_states)
    ]

    def evaluate(delta_norm: np.ndarray) -> tuple[float, list[dict[str, float]]]:
        projected = _project_linf_masked_np(delta_norm, epsilon, mask_np)
        return _closed_loop_score(
            model,
            stats,
            attack_trajs,
            initial_states,
            clean_metrics,
            robot_params,
            sim_params,
            projected,
            score_mode,
        )

    candidates = [np.zeros(input_dim, dtype=np.float32)]
    for dim in active_dims:
        for sign in (-1.0, 1.0):
            candidate = np.zeros(input_dim, dtype=np.float32)
            candidate[dim] = sign * epsilon
            candidates.append(candidate)
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            candidate = np.zeros(input_dim, dtype=np.float32)
            for dim, sign in zip(active_dims[:2], (sign_x, sign_y)):
                candidate[dim] = sign * epsilon
            candidates.append(candidate)

    best_delta = candidates[0]
    best_score, best_attacked_metrics = evaluate(best_delta)
    evaluations = 1
    for candidate in candidates[1:]:
        score, attacked_metrics = evaluate(candidate)
        evaluations += 1
        if score > best_score:
            best_score = score
            best_delta = _project_linf_masked_np(candidate, epsilon, mask_np)
            best_attacked_metrics = attacked_metrics

    max_steps = max(1, int(steps))
    population = max(4, int(population_size))
    for step_idx in range(max_steps):
        sigma = epsilon * max(0.05, 0.55 * (1.0 - step_idx / max_steps))
        improved = False
        for _ in range(population):
            noise = np.zeros(input_dim, dtype=np.float32)
            noise[active_dims] = rng.normal(0.0, sigma, size=active_dims.size).astype(np.float32)
            candidate = _project_linf_masked_np(best_delta + noise, epsilon, mask_np)
            score, attacked_metrics = evaluate(candidate)
            evaluations += 1
            if score > best_score:
                best_score = score
                best_delta = candidate
                best_attacked_metrics = attacked_metrics
                improved = True
        if log_progress:
            raw = best_delta * stats["x_std"].astype(np.float32)
            print(
                f"[attack] closed-loop step {step_idx + 1:02d}/{max_steps:02d} "
                f"score={best_score:.6f} raw_xy=({raw[0]:+.5f}, {raw[1]:+.5f}) "
                f"improved={improved}",
                flush=True,
            )

    delta_norm_np = _project_linf_masked_np(best_delta, epsilon, mask_np)
    delta_raw_np = delta_norm_np * stats["x_std"].astype(np.float32)
    metadata = {
        "attack_method": "closed_loop_blackbox",
        "optimizer": "blackbox_random_population",
        "score_mode": score_mode,
        "score": float(best_score),
        "epsilon": float(epsilon),
        "steps": int(max_steps),
        "horizon_steps": None if horizon_steps is None else int(horizon_steps),
        "population_size": int(population),
        "evaluations": int(evaluations),
        "clean_metrics": clean_metrics,
        "attacked_metrics": best_attacked_metrics,
        "optimization_clean_metrics": clean_metrics,
        "optimization_attacked_metrics": best_attacked_metrics,
    }
    return delta_raw_np.astype(np.float32), delta_norm_np.astype(np.float32), metadata


def learn_closed_loop_uap(
    model: WheelMLP,
    stats: dict[str, np.ndarray],
    trajectories: list[Trajectory],
    initial_states: list[np.ndarray],
    robot_params: RobotParams,
    sim_params: SimParams,
    epsilon: float,
    steps: int,
    population_size: int | None = None,
    seed: int = 0,
    mask: np.ndarray | None = None,
    horizon_steps: int | None = None,
    score_mode: str = "mean",
    log_progress: bool = False,
    optimizer: str = "blackbox",
    lr: float | None = None,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Dispatch closed-loop UAP optimization by optimizer name."""
    if mask is None:
        mask = np.ones(int(stats["x_std"].shape[0]), dtype=np.float32)
    if optimizer == "blackbox":
        return learn_closed_loop_uap_blackbox(
            model,
            stats,
            trajectories,
            initial_states,
            robot_params,
            sim_params,
            epsilon=epsilon,
            steps=steps,
            population_size=4 if population_size is None else population_size,
            seed=seed,
            mask=mask,
            horizon_steps=horizon_steps,
            score_mode=score_mode,
            log_progress=log_progress,
        )
    if optimizer == "gradient":
        return learn_closed_loop_uap_gradient(
            model,
            stats,
            trajectories,
            initial_states,
            robot_params,
            sim_params,
            epsilon=epsilon,
            steps=steps,
            lr=max(float(epsilon) / 8.0, 1e-4) if lr is None else lr,
            mask=mask,
            horizon_steps=horizon_steps,
            score_mode=score_mode,
            log_progress=log_progress,
            device=device,
        )
    raise ValueError(f"Unsupported closed-loop UAP optimizer: {optimizer}")
