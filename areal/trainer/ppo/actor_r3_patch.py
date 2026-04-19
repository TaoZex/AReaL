# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MoE routing metrics and R3 logging helpers for the PPO actor.

Provides two categories of metrics:

1. **R3 data stats** (``log_r3_data_stats``): Summary of the routed_experts
   tensor shape, dtype, and basic coverage info.  Logged when R3 is enabled.

2. **MoE routing effectiveness metrics** (``log_moe_routing_metrics``):
   SkyRL-style routing quality indicators that are useful for ANY MoE model,
   regardless of whether R3 is enabled.  These include:
   - Routing entropy (per-layer and aggregated)
   - Expert utilization balance (std dev of expert load)
   - Data coverage ratio (fraction of samples with valid routing data)
   - Top-1 expert concentration (how much traffic goes to most-used expert)
   - Expert diversity (number of unique experts used per token)

The key R3-specific effectiveness metrics are:

1. **Router Agreement Rate** -- fraction of tokens where training routing
   matches the replayed (inference-time) routing. Measures how effectively
   R3 forces routing alignment.

2. **Per-Layer Routing Entropy** -- Shannon entropy of the expert probability
   distribution per MoE layer. Lower entropy under replay indicates stronger
   routing concentration (expected when replay overrides natural routing).

3. **Expert Utilization Balance** -- standard deviation of per-expert token
   counts normalised by the mean. High balance (low std/mean) indicates
   evenly distributed expert usage; replay may skew this.

4. **Routing Data Coverage** -- fraction of micro-batches that carried valid
   replay data. Should be 1.0 in a healthy R3 run.

All logging uses the ``stats_tracker`` infrastructure so that metrics
appear in the same TensorBoard / WandB dashboards as other PPO stats.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from areal.utils import stats_tracker

logger = logging.getLogger(__name__)


def _ensure_routed_experts_tensor(
    data: dict[str, Any],
) -> torch.Tensor | None:
    """Retrieve ``routed_experts`` from *data* and ensure it is a ``torch.Tensor``.

    If the value is a ``np.ndarray``, ``RTensor``, or another array-like, it
    is converted in-place so that all downstream code sees a proper tensor.

    Returns the tensor or ``None`` if the key is missing / conversion fails.
    """
    re = data.get("routed_experts")
    if re is None:
        return None

    if isinstance(re, torch.Tensor):
        return re

    # --- Defensive conversion for non-tensor values ---
    original_type = type(re).__name__
    try:
        # RTensor: AReaL's remote tensor wrapper used by the RPC layer.
        # After all_gather / broadcast, tensors may arrive as RTensor
        # whose real data must be fetched via .to_local().
        if hasattr(re, "to_local") and callable(re.to_local):
            converted = re.to_local()
            if not isinstance(converted, torch.Tensor):
                converted = torch.tensor(converted, dtype=torch.int32)
            elif converted.dtype not in (torch.int32, torch.int64, torch.uint8, torch.int16):
                converted = converted.to(torch.int32)
        elif isinstance(re, np.ndarray):
            converted = torch.from_numpy(re.astype(np.int32))
        else:
            converted = torch.tensor(re, dtype=torch.int32)
        data["routed_experts"] = converted
        logger.info(
            "[R3] _ensure_routed_experts_tensor: converted routed_experts from "
            "%s (shape=%s) to torch.Tensor (shape=%s, dtype=%s).",
            original_type,
            getattr(re, "shape", "N/A"),
            converted.shape,
            converted.dtype,
        )
        return converted
    except Exception:
        logger.warning(
            "[R3] _ensure_routed_experts_tensor: failed to convert "
            "routed_experts (type=%s) to torch.Tensor; discarding.",
            original_type,
            exc_info=True,
        )
        data.pop("routed_experts", None)
        return None



def _localize_rtensors_in_dict(d: dict[str, Any]) -> None:
    """Convert any ``RTensor`` values in *d* to real ``torch.Tensor`` in-place.

    After distributed all-gather / broadcast, trajectory dicts may contain
    ``RTensor`` wrappers (AReaL's remote tensor type) instead of plain
    ``torch.Tensor`` objects.  Downstream code (``concat_padded_tensors``,
    logging helpers, etc.) expects real tensors, so we localize them here.

    Uses duck-typing (``hasattr(v, "to_local")``) to avoid a hard import
    dependency on the RPC module.
    """
    for key, val in list(d.items()):
        if hasattr(val, "to_local") and callable(val.to_local):
            try:
                d[key] = val.to_local()
            except Exception:
                logger.warning(
                    "[R3] _localize_rtensors_in_dict: failed to localize "
                    "key=%r (type=%s); leaving as-is.",
                    key, type(val).__name__, exc_info=True,
                )


def normalize_routed_experts_keys(
    trajs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure every trajectory dict in *trajs* has a consistent set of keys
    with respect to ``routed_experts``, and that all values are proper
    ``torch.Tensor`` objects.

    After rollout redistribution, trajectory dicts may exhibit two issues:

    1. **Key mismatch** -- some dicts carry ``routed_experts`` while others
       do not (inference abort, ``extract_routed_experts`` returning ``None``).
    2. **Type mismatch** -- values may be ``RTensor`` (AReaL's remote tensor
       wrapper) or ``np.ndarray`` instead of ``torch.Tensor``.  Both
       ``concat_padded_tensors`` / ``_pad_cat_dim0`` and logging helpers
       (e.g. ``attn_mask.bool()``) require real tensors.

    Strategy
    --------
    * First, localize **all** RTensor values in every dict (not just
      ``routed_experts``).  This prevents downstream crashes in code that
      accesses ``attention_mask``, ``input_ids``, etc.
    * Then convert ``routed_experts`` specifically via
      ``_ensure_routed_experts_tensor`` (handles ``np.ndarray`` and other
      non-RTensor array-likes).  Failed conversions cause the key to be
      popped.
    * Finally, reconcile key presence: pad missing ``routed_experts``
      entries with zero tensors of matching shape.

    This function mutates *trajs* **in-place** and also returns the list
    for convenience.
    """
    if not trajs:
        return trajs

    # --- Phase 1: RTensor localisation + type normalisation ---
    # Convert ALL RTensor values in every dict to real torch.Tensor,
    # then specifically handle routed_experts (np.ndarray, etc.).
    for d in trajs:
        _localize_rtensors_in_dict(d)
        if "routed_experts" in d:
            _ensure_routed_experts_tensor(d)

    # --- Phase 2: Key reconciliation ---
    has_key = [("routed_experts" in d) for d in trajs]
    n_has = sum(has_key)

    # All consistent -- nothing more to do
    if n_has == 0 or n_has == len(trajs):
        return trajs

    # Find a reference tensor to derive (num_moe_layers, topk)
    ref_tensor: torch.Tensor | None = None
    for d, hk in zip(trajs, has_key):
        if hk:
            t = d.get("routed_experts")
            if isinstance(t, torch.Tensor) and t.dim() == 4:
                ref_tensor = t
                break

    if ref_tensor is None:
        # Cannot determine shape -- strip key from all dicts so they are
        # consistent (safe fallback: training proceeds without R3 data).
        logger.warning(
            "[R3] normalize_routed_experts_keys: %d/%d trajs have "
            "routed_experts but none yielded a valid 4-D tensor; "
            "stripping key from all trajs.",
            n_has, len(trajs),
        )
        for d in trajs:
            d.pop("routed_experts", None)
        return trajs

    num_moe_layers = ref_tensor.shape[2]
    topk = ref_tensor.shape[3]
    ref_dtype = ref_tensor.dtype
    ref_device = ref_tensor.device

    n_padded = 0
    for i, (d, hk) in enumerate(zip(trajs, has_key)):
        if not hk:
            # Derive seq_len from this dict's own attention_mask
            attn = d.get("attention_mask")
            if attn is not None:
                batch_size = attn.shape[0]
                seq_len = attn.shape[-1]
            else:
                # Fallback: use input_ids shape
                ids = d.get("input_ids")
                if ids is not None:
                    batch_size = ids.shape[0]
                    seq_len = ids.shape[-1]
                else:
                    batch_size = 1
                    seq_len = 1

            d["routed_experts"] = torch.zeros(
                batch_size, seq_len, num_moe_layers, topk,
                dtype=ref_dtype, device=ref_device,
            )
            n_padded += 1

    logger.info(
        "[R3] normalize_routed_experts_keys: %d/%d trajs were missing "
        "routed_experts; padded with zeros (shape template: "
        "num_moe_layers=%d, topk=%d, dtype=%s).",
        n_padded, len(trajs), num_moe_layers, topk, ref_dtype,
    )
    return trajs


def log_r3_data_stats(
    data: dict[str, Any],
    scope: str = "r3",
) -> None:
    """Log summary statistics about the ``routed_experts`` tensor in a
    training data dict.

    Called once per PPO update step (not per micro-batch) to avoid
    log spam.

    Args:
        data: The training data dict that may contain ``"routed_experts"``.
        scope: Stats-tracker scope prefix.
    """
    re = _ensure_routed_experts_tensor(data)
    if re is None:
        return

    with stats_tracker.scope(scope):
        if isinstance(re, torch.Tensor):
            stats_tracker.scalar(
                r3_present=1,
                r3_batch_size=re.shape[0],
                r3_seq_len=re.shape[1],
                r3_num_layers=re.shape[2] if re.dim() >= 3 else 0,
                r3_topk=re.shape[3] if re.dim() >= 4 else 0,
                r3_dtype_bytes=re.element_size(),
                r3_max_expert_id=re.max().item() if re.numel() > 0 else 0,
            )

            # Compute R3 effectiveness metrics
            _log_r3_effectiveness_metrics(re)
        else:
            logger.warning(
                "[R3] log_r3_data_stats: routed_experts is %s after "
                "ensure_tensor; this should not happen.",
                type(re).__name__,
            )
            stats_tracker.scalar(r3_present=0)


def split_routed_experts_for_minibatches(
    routed_experts: torch.Tensor,
    mb_inputs,
) -> list[torch.Tensor | None]:
    """Split the global ``routed_experts`` tensor into per-mini-batch slices.

    The actor's ``_ppo_update`` pops ``routed_experts`` from the data dict
    *before* splitting into mini-batches (because the 4-D tensor does not
    fit through ``pack_tensor_dict``).  This function performs the
    corresponding split so that each mini-batch gets its own slice.

    Args:
        routed_experts: ``(bs, seq_len, num_moe_layers, topk)`` tensor.
        mb_inputs: ``MicroBatchList`` or iterable of micro-batch dicts
            that tells us how many samples each mini-batch has.

    Returns:
        List of tensors (one per mini-batch) or ``None`` entries if the
        mini-batch has no routing data.
    """
    if routed_experts is None:
        return []

    total_bs = routed_experts.shape[0]

    # Determine per-mini-batch sample counts
    if hasattr(mb_inputs, "__len__"):
        n_mbs = len(mb_inputs)
    else:
        n_mbs = 1

    mb_sizes = []
    if hasattr(mb_inputs, "mbs") and mb_inputs.mbs:
        for mb_dict in mb_inputs.mbs:
            mb_sizes.append(_infer_mb_sample_count(mb_dict, total_bs, n_mbs))
    elif hasattr(mb_inputs, "__iter__"):
        for mb_item in mb_inputs:
            mb_dict = mb_item if isinstance(mb_item, dict) else getattr(mb_item, "orig_mb", mb_item)
            mb_sizes.append(_infer_mb_sample_count(mb_dict, total_bs, n_mbs))
    else:
        # Fallback: equal split
        base = total_bs // n_mbs
        remainder = total_bs % n_mbs
        mb_sizes = [base + (1 if i < remainder else 0) for i in range(n_mbs)]

    # Validate
    if sum(mb_sizes) != total_bs:
        logger.warning(
            "[R3] split_routed_experts_for_minibatches: sum(mb_sizes)=%d != "
            "total_bs=%d. Falling back to equal split.",
            sum(mb_sizes), total_bs,
        )
        base = total_bs // n_mbs
        remainder = total_bs % n_mbs
        mb_sizes = [base + (1 if i < remainder else 0) for i in range(n_mbs)]

    splits = torch.split(routed_experts, mb_sizes, dim=0)
    return list(splits)


def _infer_mb_sample_count(
    mb_dict: Any,
    total_bs: int,
    n_mbs: int,
) -> int:
    """Infer sample count from a mini-batch dict."""
    if isinstance(mb_dict, dict):
        attn = mb_dict.get("attention_mask")
        if attn is not None and hasattr(attn, "shape"):
            return attn.shape[0]
        ids = mb_dict.get("input_ids")
        if ids is not None and hasattr(ids, "shape"):
            return ids.shape[0]
    return total_bs // n_mbs


def _log_r3_effectiveness_metrics(
    routed_experts: torch.Tensor,
) -> None:
    """Compute and log R3 effectiveness metrics following SkyRL's approach.

    These metrics help assess whether Router Replay is working correctly
    and how it affects the MoE routing distribution.

    Args:
        routed_experts: ``(bs, seq_len, num_moe_layers, topk)`` int tensor
            containing the expert indices from inference.
    """
    if routed_experts.dim() != 4 or routed_experts.numel() == 0:
        return

    bs, seq_len, num_moe_layers, topk = routed_experts.shape

    try:
        # --- Metric 1: Per-Layer Routing Entropy ---
        # Measures the diversity of expert assignments per layer.
        # Lower entropy = more concentrated routing.
        # Under R3, this reflects the inference-time routing distribution.
        _log_per_layer_routing_entropy(routed_experts, num_moe_layers, topk)

        # --- Metric 2: Expert Utilization Balance ---
        # Measures how evenly tokens are distributed across experts.
        # Coefficient of variation (std/mean) -- lower = more balanced.
        _log_expert_utilization_balance(routed_experts, num_moe_layers)

        # --- Metric 3: Routing Data Coverage ---
        # Fraction of (batch, layer) combinations with non-zero routing data.
        _log_routing_data_coverage(routed_experts, bs, num_moe_layers)

        # --- Metric 4: Top-1 Expert Concentration ---
        # How often the most popular expert is selected (per layer).
        _log_top1_expert_concentration(routed_experts, num_moe_layers)

    except Exception:
        logger.warning(
            "[R3] Failed to compute R3 effectiveness metrics.",
            exc_info=True,
        )


def _log_per_layer_routing_entropy(
    routed_experts: torch.Tensor,
    num_moe_layers: int,
    topk: int,
) -> None:
    """Log per-layer Shannon entropy of expert routing distribution.

    For each MoE layer, computes the probability distribution over experts
    (from the replay data) and its Shannon entropy.  Reports mean, min,
    max across layers.
    """
    bs, seq_len = routed_experts.shape[:2]
    # Flatten batch and seq dimensions
    flat = routed_experts.view(-1, num_moe_layers, topk)  # (bs*seq_len, L, K)
    num_tokens = flat.shape[0]

    if num_tokens == 0:
        return

    # Determine number of experts from max index
    num_experts = int(routed_experts.max().item()) + 1
    if num_experts <= 0:
        return

    layer_entropies = []
    for layer_idx in range(num_moe_layers):
        # Count expert occurrences for this layer across all tokens and topk slots
        expert_ids = flat[:, layer_idx, :].reshape(-1).long()
        # Filter out padding (expert_id == 0 might be valid, but -1 or very large is not)
        valid_mask = (expert_ids >= 0) & (expert_ids < num_experts)
        expert_ids = expert_ids[valid_mask]
        if expert_ids.numel() == 0:
            continue

        counts = torch.bincount(expert_ids, minlength=num_experts).float()
        probs = counts / counts.sum()
        # Shannon entropy: -sum(p * log(p)), with 0*log(0) = 0
        log_probs = torch.where(probs > 0, torch.log2(probs), torch.zeros_like(probs))
        entropy = -(probs * log_probs).sum().item()
        layer_entropies.append(entropy)

    if layer_entropies:
        mean_entropy = sum(layer_entropies) / len(layer_entropies)
        min_entropy = min(layer_entropies)
        max_entropy = max(layer_entropies)
        # Maximum possible entropy for reference
        max_possible = torch.log2(torch.tensor(float(num_experts))).item()

        stats_tracker.scalar(
            r3_routing_entropy_mean=mean_entropy,
            r3_routing_entropy_min=min_entropy,
            r3_routing_entropy_max=max_entropy,
            r3_routing_entropy_normalised=mean_entropy / max_possible if max_possible > 0 else 0,
            r3_num_experts=num_experts,
        )


def _log_expert_utilization_balance(
    routed_experts: torch.Tensor,
    num_moe_layers: int,
) -> None:
    """Log expert utilization balance (coefficient of variation per layer).

    For each layer, compute the standard deviation of per-expert token
    counts divided by the mean.  Aggregate across layers.
    """
    flat = routed_experts.view(-1, num_moe_layers, routed_experts.shape[-1])
    num_experts = int(routed_experts.max().item()) + 1
    if num_experts <= 1:
        return

    layer_cv_values = []
    for layer_idx in range(num_moe_layers):
        expert_ids = flat[:, layer_idx, :].reshape(-1).long()
        valid_mask = (expert_ids >= 0) & (expert_ids < num_experts)
        expert_ids = expert_ids[valid_mask]
        if expert_ids.numel() == 0:
            continue

        counts = torch.bincount(expert_ids, minlength=num_experts).float()
        mean_count = counts.mean()
        if mean_count > 0:
            cv = counts.std() / mean_count
            layer_cv_values.append(cv.item())

    if layer_cv_values:
        stats_tracker.scalar(
            r3_expert_util_cv_mean=sum(layer_cv_values) / len(layer_cv_values),
            r3_expert_util_cv_max=max(layer_cv_values),
            r3_expert_util_cv_min=min(layer_cv_values),
        )


def _log_routing_data_coverage(
    routed_experts: torch.Tensor,
    bs: int,
    num_moe_layers: int,
) -> None:
    """Log fraction of (sample, layer) with non-zero routing data."""
    # Check each sample x layer has at least one non-zero expert id
    # routed_experts: (bs, seq_len, num_moe_layers, topk)
    # Sum over seq_len and topk dimensions
    has_data = (routed_experts.sum(dim=(1, 3)) > 0).float()  # (bs, num_moe_layers)
    coverage = has_data.mean().item()
    stats_tracker.scalar(r3_routing_data_coverage=coverage)


def _log_top1_expert_concentration(
    routed_experts: torch.Tensor,
    num_moe_layers: int,
) -> None:
    """Log how concentrated routing is on the most popular expert per layer.

    For each layer, the concentration ratio = count(most_popular_expert) / total_count.
    High concentration suggests the replay data has strong routing preferences.
    """
    flat = routed_experts.view(-1, num_moe_layers, routed_experts.shape[-1])
    num_experts = int(routed_experts.max().item()) + 1
    if num_experts <= 0:
        return

    layer_concentrations = []
    for layer_idx in range(num_moe_layers):
        expert_ids = flat[:, layer_idx, :].reshape(-1).long()
        valid_mask = (expert_ids >= 0) & (expert_ids < num_experts)
        expert_ids = expert_ids[valid_mask]
        if expert_ids.numel() == 0:
            continue

        counts = torch.bincount(expert_ids, minlength=num_experts)
        max_count = counts.max().item()
        total = counts.sum().item()
        if total > 0:
            layer_concentrations.append(max_count / total)

    if layer_concentrations:
        stats_tracker.scalar(
            r3_top1_expert_concentration_mean=sum(layer_concentrations) / len(layer_concentrations),
            r3_top1_expert_concentration_max=max(layer_concentrations),
        )


def compute_router_agreement_rate(
    replay_indices: torch.Tensor,
    actual_indices: torch.Tensor,
) -> float:
    """Compute the fraction of tokens where actual routing matches replay target.

    This is the KEY R3 effectiveness metric: if R3 is working correctly,
    agreement should be very close to 1.0 (training router produces the same
    assignments as the replayed inference routing).

    Args:
        replay_indices: ``(num_tokens, topk)`` target expert indices from replay.
        actual_indices: ``(num_tokens, topk)`` actual expert indices from training.

    Returns:
        Agreement rate in [0, 1].  Returns -1.0 if inputs are invalid.
    """
    if replay_indices is None or actual_indices is None:
        return -1.0
    if replay_indices.shape != actual_indices.shape:
        logger.warning(
            "[R3] Agreement rate: shape mismatch replay=%s vs actual=%s.",
            replay_indices.shape, actual_indices.shape,
        )
        return -1.0

    # Sort topk indices per token to handle different ordering
    replay_sorted = replay_indices.sort(dim=-1).values
    actual_sorted = actual_indices.sort(dim=-1).values
    matches = (replay_sorted == actual_sorted).all(dim=-1).float()
    return matches.mean().item()


def log_router_agreement_rate(
    replay_indices: torch.Tensor,
    actual_indices: torch.Tensor,
    scope: str = "r3",
) -> None:
    """Compute and log router agreement rate to stats_tracker / wandb.

    This is the **core R3 differential metric**: it directly measures how
    successfully R3 forces the training-time router to match inference-time
    routing decisions.

    When comparing R3-enabled vs non-R3 runs:
    - R3 enabled: agreement rate should approach 1.0
    - R3 disabled: agreement rate reflects natural train/inference divergence

    Args:
        replay_indices: ``(num_tokens, topk)`` target expert indices (replay).
        actual_indices: ``(num_tokens, topk)`` actual expert indices (training).
        scope: Stats-tracker scope prefix.
    """
    rate = compute_router_agreement_rate(replay_indices, actual_indices)
    if rate < 0:
        return

    with stats_tracker.scope(scope):
        stats_tracker.scalar(router_agreement_rate=rate)

    # Also compute per-layer agreement if we can reshape
    # This requires knowledge of num_moe_layers which is not available here;
    # see log_per_layer_router_agreement for the per-layer variant.
    logger.info("[R3] Router agreement rate: %.4f", rate)


def log_per_layer_router_agreement(
    replay_indices: torch.Tensor,
    actual_indices: torch.Tensor,
    num_moe_layers: int,
    scope: str = "r3",
) -> None:
    """Compute and log per-layer router agreement rate.

    Useful for identifying which MoE layers have the most routing divergence
    between inference and training.

    Args:
        replay_indices: ``(num_tokens, num_moe_layers, topk)`` replay routing.
        actual_indices: ``(num_tokens, num_moe_layers, topk)`` actual routing.
        num_moe_layers: Number of MoE layers.
        scope: Stats-tracker scope prefix.
    """
    if replay_indices is None or actual_indices is None:
        return
    if replay_indices.shape != actual_indices.shape:
        return

    with stats_tracker.scope(scope):
        layer_rates = []
        for layer_idx in range(num_moe_layers):
            r = replay_indices[:, layer_idx, :]
            a = actual_indices[:, layer_idx, :]
            rate = compute_router_agreement_rate(r, a)
            if rate >= 0:
                layer_rates.append(rate)
                stats_tracker.scalar(
                    **{f"router_agreement_rate_layer_{layer_idx}": rate}
                )

        if layer_rates:
            stats_tracker.scalar(
                router_agreement_rate_mean=sum(layer_rates) / len(layer_rates),
                router_agreement_rate_min=min(layer_rates),
                router_agreement_rate_max=max(layer_rates),
            )


def log_r3_vs_baseline_metrics(
    data: dict[str, Any],
    scope: str = "r3_comparison",
) -> None:
    """Log comprehensive metrics for comparing R3-enabled vs non-R3 (baseline) runs.

    These metrics are designed so that when two WandB runs (one with R3, one
    without) are overlaid, the differences clearly show the impact of R3:

    **Metrics logged (all under *scope*):**

    - ``routing_consistency_rate``: Fraction of tokens with stable routing
      across the batch (higher = more consistent routing, R3 should increase).
    - ``routing_entropy_per_layer``: Mean normalised entropy (R3 may decrease
      as replay concentrates routing).
    - ``expert_load_balance_score``: 1 - CV of expert counts (higher = more
      balanced; R3 should maintain or improve).
    - ``expert_coverage_ratio``: Fraction of experts that receive > 0 tokens.
    - ``routing_confidence``: Mean top-1 probability proxy (ratio of top-1
      count to total, higher under R3 = stronger routing decisions).
    - ``batch_routing_diversity``: Average number of unique experts per layer
      across the batch (R3 may reduce diversity if replay concentrates).
    - ``cross_sample_routing_agreement``: Pairwise agreement between samples
      in the batch for the same token position (R3 should increase for
      deterministic prompts).

    Args:
        data: Training data dict with ``"routed_experts"`` of shape
            ``(bs, seq_len, num_moe_layers, topk)`` and ``"attention_mask"``.
        scope: Stats-tracker scope prefix.
    """
    re = _ensure_routed_experts_tensor(data)
    if re is None or not isinstance(re, torch.Tensor) or re.dim() != 4:
        return

    bs, seq_len, num_layers, topk = re.shape
    attn_mask = data.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.bool()
        attn_seq = attn_mask.shape[-1]
        if attn_seq < seq_len:
            # attention_mask was trimmed but routed_experts retains padded length;
            # extend mask with False (padding positions).
            real_mask = torch.zeros(bs, seq_len, dtype=torch.bool, device=re.device)
            real_mask[:, :attn_seq] = attn_mask
        elif attn_seq > seq_len:
            real_mask = attn_mask[:, :seq_len]
        else:
            real_mask = attn_mask
    else:
        real_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=re.device)

    num_experts = int(re.max().item()) + 1
    if num_experts < 2:
        return

    with stats_tracker.scope(scope):
        # ---- 1. Routing Consistency Rate ----
        # For each layer, measure how often the top-1 expert is the same
        # across the topk slots (proxy: first slot consistency across batch).
        _log_routing_consistency_rate(re, real_mask, num_layers, num_experts)

        # ---- 2. Expert Load Balance Score ----
        _log_expert_load_balance_score(re, real_mask, num_layers, num_experts)

        # ---- 3. Expert Coverage Ratio ----
        _log_expert_coverage_ratio(re, real_mask, num_layers, num_experts)

        # ---- 4. Routing Confidence ----
        _log_routing_confidence(re, real_mask, num_layers, num_experts)

        # ---- 5. Batch Routing Diversity ----
        _log_batch_routing_diversity(re, real_mask, num_layers, num_experts)

        # ---- 6. Cross-Sample Routing Agreement ----
        if bs >= 2:
            _log_cross_sample_routing_agreement(re, real_mask, num_layers)


def _log_routing_consistency_rate(
    re: torch.Tensor,
    real_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> None:
    """Fraction of tokens where the top-1 expert is consistently the mode expert.

    For each (layer, position), find the mode expert across the batch.
    Consistency = fraction of samples that agree with the mode.
    Higher consistency under R3 indicates successful routing alignment.
    """
    bs, seq_len = re.shape[:2]
    consistency_rates = []

    for layer_idx in range(num_layers):
        top1 = re[:, :, layer_idx, 0].long()  # (bs, seq_len)
        valid = real_mask  # (bs, seq_len)

        # For each position, find the mode across the batch
        for pos in range(0, min(seq_len, 128), 4):  # Sample positions to avoid O(n^2)
            col = top1[:, pos]
            mask = valid[:, pos]
            col = col[mask]
            if col.numel() < 2:
                continue
            mode_val = torch.bincount(col.clamp(0, num_experts - 1)).argmax()
            agree = (col == mode_val).float().mean().item()
            consistency_rates.append(agree)

    if consistency_rates:
        stats_tracker.scalar(
            routing_consistency_rate=sum(consistency_rates) / len(consistency_rates),
        )


def _log_expert_load_balance_score(
    re: torch.Tensor,
    real_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> None:
    """1 - CV of expert assignment counts (higher = more balanced)."""
    balance_scores = []
    token_mask = real_mask.unsqueeze(-1).unsqueeze(-1).expand_as(re)

    for layer_idx in range(num_layers):
        layer_re = re[:, :, layer_idx, :]
        layer_mask = real_mask.unsqueeze(-1).expand_as(layer_re)
        valid_experts = layer_re[layer_mask].long().clamp(0, num_experts - 1)
        if valid_experts.numel() == 0:
            continue
        counts = torch.bincount(valid_experts, minlength=num_experts).float()
        mean_c = counts.mean()
        if mean_c > 0:
            cv = (counts.std() / mean_c).item()
            balance_scores.append(max(0.0, 1.0 - cv))

    if balance_scores:
        stats_tracker.scalar(
            expert_load_balance_score=sum(balance_scores) / len(balance_scores),
        )


def _log_expert_coverage_ratio(
    re: torch.Tensor,
    real_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> None:
    """Fraction of experts that receive at least 1 token."""
    coverage_ratios = []
    for layer_idx in range(num_layers):
        layer_re = re[:, :, layer_idx, :]
        layer_mask = real_mask.unsqueeze(-1).expand_as(layer_re)
        valid_experts = layer_re[layer_mask].long().clamp(0, num_experts - 1)
        if valid_experts.numel() == 0:
            continue
        counts = torch.bincount(valid_experts, minlength=num_experts)
        active = (counts > 0).sum().item()
        coverage_ratios.append(active / num_experts)

    if coverage_ratios:
        stats_tracker.scalar(
            expert_coverage_ratio=sum(coverage_ratios) / len(coverage_ratios),
        )


def _log_routing_confidence(
    re: torch.Tensor,
    real_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> None:
    """Routing confidence: how dominant the top-1 expert is per layer.

    For each layer, compute: max(expert_prob) across all experts.
    Higher confidence under R3 = replay is successfully concentrating routing.
    """
    confidences = []
    for layer_idx in range(num_layers):
        layer_re = re[:, :, layer_idx, :]
        layer_mask = real_mask.unsqueeze(-1).expand_as(layer_re)
        valid_experts = layer_re[layer_mask].long().clamp(0, num_experts - 1)
        if valid_experts.numel() == 0:
            continue
        counts = torch.bincount(valid_experts, minlength=num_experts).float()
        total = counts.sum()
        if total > 0:
            confidences.append((counts.max() / total).item())

    if confidences:
        stats_tracker.scalar(
            routing_confidence=sum(confidences) / len(confidences),
        )


def _log_batch_routing_diversity(
    re: torch.Tensor,
    real_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> None:
    """Average number of unique experts used per sample per layer."""
    diversities = []
    for layer_idx in range(num_layers):
        for sample_idx in range(re.shape[0]):
            sample_re = re[sample_idx, :, layer_idx, :]  # (seq_len, topk)
            sample_mask = real_mask[sample_idx]  # (seq_len,)
            valid = sample_re[sample_mask].long().clamp(0, num_experts - 1)
            if valid.numel() == 0:
                continue
            n_unique = valid.unique().numel()
            diversities.append(n_unique / num_experts)

    if diversities:
        stats_tracker.scalar(
            batch_routing_diversity=sum(diversities) / len(diversities),
        )


def _log_cross_sample_routing_agreement(
    re: torch.Tensor,
    real_mask: torch.Tensor,
    num_layers: int,
) -> None:
    """Pairwise agreement of top-1 expert between sample pairs.

    For same-position tokens, how often do different samples in the batch
    route to the same top-1 expert? R3 should increase this for
    deterministic prompts.
    """
    bs, seq_len = re.shape[:2]
    agreements = []

    # Sample a few layer/position combinations to keep cost bounded
    sample_layers = list(range(0, num_layers, max(1, num_layers // 4)))
    sample_positions = list(range(0, min(seq_len, 64), 4))

    for layer_idx in sample_layers:
        for pos in sample_positions:
            top1 = re[:, pos, layer_idx, 0].long()  # (bs,)
            mask = real_mask[:, pos]  # (bs,)
            top1 = top1[mask]
            if top1.numel() < 2:
                continue
            # Pairwise agreement: fraction of pairs that agree
            n = top1.numel()
            expanded_a = top1.unsqueeze(1).expand(n, n)
            expanded_b = top1.unsqueeze(0).expand(n, n)
            # Exclude diagonal
            agree = (expanded_a == expanded_b).float()
            # Mean of upper triangle
            mask_upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=re.device), diagonal=1)
            if mask_upper.sum() > 0:
                agreements.append(agree[mask_upper].mean().item())

    if agreements:
        stats_tracker.scalar(
            cross_sample_routing_agreement=sum(agreements) / len(agreements),
        )


def log_moe_routing_metrics(
    data: dict[str, Any],
    scope: str = "moe_routing",
) -> None:
    """Log MoE routing effectiveness metrics for ANY MoE model.

    Computes routing quality indicators from the
    ``routed_experts`` tensor.  These metrics help diagnose routing
    quality issues (expert collapse, load imbalance, etc.) and are
    useful even without R3.

    Args:
        data: Training data dict containing ``"routed_experts"``
            of shape ``(bs, seq_len, num_moe_layers, topk)``.
        scope: Stats-tracker scope prefix.
    """
    re = _ensure_routed_experts_tensor(data)
    if re is None:
        return
    if not isinstance(re, torch.Tensor) or re.dim() < 4:
        return

    bs, seq_len, num_layers, topk = re.shape
    attn_mask = data.get("attention_mask")

    with stats_tracker.scope(scope):
        # ------------------------------------------------------------------
        # 1. Data coverage: fraction of samples with non-zero routing data
        # ------------------------------------------------------------------
        has_routing = (re.sum(dim=(1, 2, 3)) != 0).float()
        coverage = has_routing.mean().item()
        stats_tracker.scalar(data_coverage=coverage)

        # ------------------------------------------------------------------
        # 2. Expert utilization and load balance (per-layer)
        # ------------------------------------------------------------------
        if attn_mask is not None:
            attn_mask_bool = attn_mask.bool()
            attn_seq = attn_mask_bool.shape[-1]
            if attn_seq < seq_len:
                # attention_mask was trimmed but routed_experts retains padded
                # length; extend mask with False (padding positions).
                real_mask = torch.zeros(bs, seq_len, dtype=torch.bool, device=re.device)
                real_mask[:, :attn_seq] = attn_mask_bool
            elif attn_seq > seq_len:
                real_mask = attn_mask_bool[:, :seq_len]
            else:
                real_mask = attn_mask_bool
        else:
            real_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=re.device)

        # Expand mask for layers and topk: (bs, seq_len, 1, 1)
        token_mask = real_mask.unsqueeze(-1).unsqueeze(-1).expand_as(re)
        max_expert_id = re[token_mask].max().item() if token_mask.any() else 0
        num_experts = int(max_expert_id) + 1
        if num_experts < 2:
            stats_tracker.scalar(
                num_experts=num_experts,
                insufficient_data=1,
            )
            return

        entropy_sum = 0.0
        balance_sum = 0.0
        top1_concentration_sum = 0.0
        diversity_sum = 0.0
        valid_layers = 0

        for layer_idx in range(num_layers):
            layer_re = re[:, :, layer_idx, :]
            layer_mask = real_mask.unsqueeze(-1).expand_as(layer_re)
            valid_experts = layer_re[layer_mask]

            if valid_experts.numel() == 0:
                continue

            valid_layers += 1

            expert_counts = torch.bincount(
                valid_experts.long().clamp(0, num_experts - 1),
                minlength=num_experts,
            ).float()
            total_assignments = expert_counts.sum()

            if total_assignments == 0:
                continue

            expert_probs = expert_counts / total_assignments

            # Routing entropy (normalized)
            log_probs = torch.log(expert_probs + 1e-10)
            entropy = -(expert_probs * log_probs).sum().item()
            max_entropy = torch.log(torch.tensor(float(num_experts))).item()
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            entropy_sum += normalized_entropy

            # Expert load imbalance (CV)
            load_std = expert_probs.std().item()
            load_mean = expert_probs.mean().item()
            balance = load_std / (load_mean + 1e-10)
            balance_sum += balance

            # Top-1 expert concentration
            top1_ratio = expert_probs.max().item()
            top1_concentration_sum += top1_ratio

            # Expert diversity
            unique_experts_used = (expert_counts > 0).sum().item()
            diversity = unique_experts_used / num_experts
            diversity_sum += diversity

        if valid_layers > 0:
            stats_tracker.scalar(
                num_experts=num_experts,
                num_moe_layers=num_layers,
                routing_entropy=entropy_sum / valid_layers,
                expert_load_imbalance_cv=balance_sum / valid_layers,
                top1_expert_concentration=top1_concentration_sum / valid_layers,
                expert_diversity=diversity_sum / valid_layers,
                valid_moe_layers=valid_layers,
            )
        else:
            stats_tracker.scalar(
                num_experts=num_experts,
                num_moe_layers=num_layers,
                valid_moe_layers=0,
            )


def strip_routed_experts_before_loss(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Remove ``routed_experts`` from the data dict before the loss function.

    The ``routed_experts`` tensor is consumed by the R3 engine patch
    during ``forward_backward_batch``, so by the time we reach the loss
    function it has already been popped.  This function is a safety net.

    Returns the data dict (modified in-place).
    """
    data.pop("routed_experts", None)
    return data
