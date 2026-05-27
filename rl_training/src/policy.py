
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class PolicyOutput:
    target_logits: torch.Tensor
    ship_bucket_logits: torch.Tensor
    value: torch.Tensor


class PlanetPolicy(nn.Module):
    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        ship_bucket_count: int = 8,
        hidden_size: int = 128,
        bucket_feature_dim: int = 4,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.ship_bucket_count = ship_bucket_count
        self.bucket_feature_dim = bucket_feature_dim
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.ship_bucket_head = nn.Sequential(
            nn.Linear(hidden_size * 3 + bucket_feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        ship_bucket_mask: torch.Tensor | None = None,
        bucket_features: torch.Tensor | None = None,
    ) -> PolicyOutput:
        self_hidden = self.self_encoder(self_features)
        global_hidden = self.global_encoder(global_features)
        candidate_hidden = self.candidate_encoder(candidate_features)
        expanded_self = self_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        expanded_global = global_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)
        target_logits = self.target_head(joint).squeeze(-1)
        target_logits = target_logits.masked_fill(~candidate_mask, torch.finfo(target_logits.dtype).min)

        B, C, H = joint.shape
        S = self.ship_bucket_count
        if bucket_features is not None and bucket_features.shape[-1] == self.bucket_feature_dim:
            joint_expanded = joint.unsqueeze(2).expand(B, C, S, H)
            bucket_input = torch.cat([joint_expanded, bucket_features], dim=-1)
            ship_bucket_logits = self.ship_bucket_head(bucket_input).squeeze(-1)
        else:
            joint_expanded = joint.unsqueeze(2).expand(B, C, S, H)
            zeros = torch.zeros(B, C, S, self.bucket_feature_dim, device=joint.device, dtype=joint.dtype)
            bucket_input = torch.cat([joint_expanded, zeros], dim=-1)
            ship_bucket_logits = self.ship_bucket_head(bucket_input).squeeze(-1)

        if ship_bucket_mask is not None:
            ship_bucket_logits = ship_bucket_logits.masked_fill(
                ~ship_bucket_mask,
                torch.finfo(ship_bucket_logits.dtype).min,
            )
        # State value is turn-level; global features already encode board position.
        value = self.value_head(global_hidden).squeeze(-1)
        return PolicyOutput(target_logits=target_logits, ship_bucket_logits=ship_bucket_logits, value=value)
