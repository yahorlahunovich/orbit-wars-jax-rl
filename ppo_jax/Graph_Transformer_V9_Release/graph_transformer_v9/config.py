"""Default config for GraphTransformerV9."""

from models.graph_transformer_v8.config import DEFAULT_CONFIG as _V8_CONFIG


DEFAULT_CONFIG = dict(_V8_CONFIG)
DEFAULT_CONFIG.update({
    "output_root": "models/graph_transformer_v9/checkpoints",
    "n_layers": 5,
    "value_layers": 4,
    "edge_dim": 16,
    "edge_input_dim": 14,
    "node_input_dim": 21,
    "max_owned": 60,
    "entropy_coef": 0.02,
    "entropy_coef_max": 0.03,
    "target_entropy_start": 0.45,
    "target_entropy_end": 0.15,
    "target_entropy_weight": 1.0,
    "frac_entropy_weight": 0.35,
    "send_logit_bias": 0.0,
    "apm_target_min": 0.3,
    "apm_target_max": 1.0,
    "apm_high_k": 3.0,
    "apm_low_k": 3.0,
    "apm_low_weight": 0.35,
    "apm_phase_bounds_enabled": True,
    "apm_phase_tick_scale": 0.6,
    "apm_penalty_coef": 0.002,
    "apm_expected_temperature": 0.3,
    "self_play_cycle_promotions": 3,
    "phase3_cycle_promotions": 1,
    "promotion_check_every": 5,
    "min_updates_between_promotions": 20,
    "high_fidelity_eval_enabled": True,
})
