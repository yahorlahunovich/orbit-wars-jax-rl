
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .policy import PolicyOutput


@dataclass(slots=True)
class SampledAction:
    target_index: torch.Tensor
    ship_bucket_index: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(slots=True)
class TransitionBatch:
    self_features: torch.Tensor
    candidate_features: torch.Tensor
    global_features: torch.Tensor
    candidate_mask: torch.Tensor
    ship_bucket_mask: torch.Tensor
    bucket_features: torch.Tensor
    target_index: torch.Tensor
    ship_bucket_index: torch.Tensor
    log_prob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def sample_actions(outputs: PolicyOutput, deterministic: bool) -> SampledAction:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_index = target_logits.argmax(dim=-1) if deterministic else target_dist.sample()
    bucket_logits = selected_bucket_logits(outputs.ship_bucket_logits, target_index)
    bucket_logits = safe_bucket_logits(bucket_logits)
    bucket_dist = Categorical(logits=bucket_logits)
    ship_bucket_index = bucket_logits.argmax(dim=-1) if deterministic else bucket_dist.sample()

    log_prob, entropy = action_log_prob_and_entropy(
        outputs=outputs,
        target_index=target_index,
        ship_bucket_index=ship_bucket_index,
    )
    return SampledAction(
        target_index=target_index,
        ship_bucket_index=ship_bucket_index,
        log_prob=log_prob,
        entropy=entropy,
    )


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor,
    ship_bucket_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_log_prob = target_dist.log_prob(target_index)
    target_entropy = target_dist.entropy()
    bucket_logits = safe_bucket_logits(selected_bucket_logits(outputs.ship_bucket_logits, target_index))
    bucket_dist = Categorical(logits=bucket_logits)
    bucket_log_prob = bucket_dist.log_prob(ship_bucket_index)
    bucket_entropy = bucket_dist.entropy()
    return target_log_prob + bucket_log_prob, target_entropy + bucket_entropy


def safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    invalid_rows = ~torch.isfinite(target_logits).any(dim=-1)
    if not invalid_rows.any():
        return target_logits
    safe_logits = target_logits.clone()
    safe_logits[invalid_rows, 0] = 0.0
    return safe_logits


def selected_bucket_logits(ship_bucket_logits: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    row_idx = torch.arange(ship_bucket_logits.shape[0], device=ship_bucket_logits.device)
    return ship_bucket_logits[row_idx, target_index]


def safe_bucket_logits(bucket_logits: torch.Tensor) -> torch.Tensor:
    invalid_rows = ~torch.isfinite(bucket_logits).any(dim=-1)
    if not invalid_rows.any():
        return bucket_logits
    safe_logits = bucket_logits.clone()
    safe_logits[invalid_rows, 0] = 0.0
    return safe_logits


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    var_returns = returns.var(unbiased=False)
    if var_returns.item() < 1e-8:
        return float("nan")
    return float(1.0 - (returns - values).var(unbiased=False) / (var_returns + 1e-8))


def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if batch.self_features.shape[0] == 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "explained_variance": float("nan"),
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
    self_features = batch.self_features.to(device)
    candidate_features = batch.candidate_features.to(device)
    global_features = batch.global_features.to(device)
    candidate_mask = batch.candidate_mask.to(device).bool()
    ship_bucket_mask = batch.ship_bucket_mask.to(device).bool()
    bucket_features = batch.bucket_features.to(device)
    old_log_prob = batch.log_prob.to(device)
    target_index = batch.target_index.to(device)
    ship_bucket_index = batch.ship_bucket_index.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    size = self_features.shape[0]
    minibatch_size = min(size, max(1, minibatch_size))
    metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "explained_variance": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    explained_variance_count = 0
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(size, device=device)
        for start in range(0, size, minibatch_size):
            idx = order[start : start + minibatch_size]
            outputs = policy(
                self_features[idx],
                candidate_features[idx],
                global_features[idx],
                candidate_mask[idx],
                ship_bucket_mask[idx],
                bucket_features[idx],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                outputs,
                target_index[idx],
                ship_bucket_index[idx],
            )
            log_ratio = new_log_prob - old_log_prob[idx]
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(
                -advantages[idx] * ratio,
                -advantages[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()
            value_loss = 0.5 * (returns[idx] - outputs.value).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
                explained_variance = _explained_variance(returns[idx], outputs.value)
            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            metrics["approx_kl"] += float(approx_kl.detach().cpu())
            metrics["clip_fraction"] += float(clip_fraction.detach().cpu())
            if not math.isnan(explained_variance):
                metrics["explained_variance"] += explained_variance
                explained_variance_count += 1
            updates += 1
    averaged = {key: value / max(updates, 1) for key, value in metrics.items()}
    if explained_variance_count > 0:
        averaged["explained_variance"] /= explained_variance_count
    else:
        averaged["explained_variance"] = float("nan")
    return averaged
