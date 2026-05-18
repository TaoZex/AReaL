# SPDX-License-Identifier: Apache-2.0
"""Multi-teacher On-Policy Distillation (MoPD) helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover
    from areal.api.cli_args import TeacherConfig


logger = logging.getLogger("areal.mopd")


DEFAULT_TEACHER_KEY: str = "data_source"


def _is_teachers_dict(teachers: Any) -> bool:
    return isinstance(teachers, Mapping) and len(teachers) > 0


def is_mopd_enabled(config: Any) -> bool:
    """Return True iff the multi-teacher OPD path should be used."""
    teachers = getattr(config, "teachers", None)
    enabled = _is_teachers_dict(teachers)
    logger.info(
        "[MoPD] is_mopd_enabled: teachers=%s -> enabled=%s",
        list(teachers.keys()) if _is_teachers_dict(teachers) else None,
        enabled,
    )
    return enabled


def resolve_teacher_key(
    sample: Mapping[str, Any],
    teacher_key: str,
    available_keys: Iterable[str],
) -> str:
    """Resolve which configured teacher a given sample routes to."""
    if teacher_key not in sample:
        raise KeyError(
            f"[MoPD] sample is missing routing field '{teacher_key}'. "
            f"Available fields: {list(sample.keys())}"
        )

    value = sample[teacher_key]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(
                f"[MoPD] sample routing field '{teacher_key}' is an empty "
                "list/tuple; cannot resolve a teacher."
            )
        first = value[0]
        if any(v != first for v in value[1:]):
            distinct = sorted({str(v) for v in value})
            raise ValueError(
                f"[MoPD] sample routing field '{teacher_key}' has "
                f"non-homogeneous list contents {distinct!r}; expected all "
                "elements to be identical (one routing key per prompt)."
            )
        value = first
    routing_key = value.decode() if isinstance(value, (bytes, bytearray)) else str(value)

    available = list(available_keys)
    if routing_key not in available:
        raise ValueError(
            f"[MoPD] sample routing value {routing_key!r} (from field "
            f"'{teacher_key}') has no matching teacher. Configured keys: "
            f"{available}"
        )
    return routing_key


def group_trajectories_by_teacher(
    rollout_batch: list[dict[str, Any]],
    teachers_cfg: Mapping[str, "TeacherConfig"],
    teacher_key: str,
) -> dict[str, list[int]]:
    """Group trajectory indices by the teacher each one routes to."""
    key_to_name: dict[str, str] = {}
    for name, t_cfg in teachers_cfg.items():
        cfg_key = getattr(t_cfg, "key", None)
        routing = name if cfg_key is None else str(cfg_key)
        if routing in key_to_name:
            raise ValueError(
                f"[MoPD] duplicate teacher routing key {routing!r} between "
                f"{key_to_name[routing]!r} and {name!r}"
            )
        key_to_name[routing] = name

    groups: dict[str, list[int]] = {name: [] for name in teachers_cfg.keys()}
    for idx, traj in enumerate(rollout_batch):
        routing_key = resolve_teacher_key(traj, teacher_key, key_to_name.keys())
        teacher_name = key_to_name[routing_key]
        groups[teacher_name].append(idx)

    logger.info(
        "[MoPD] group_trajectories_by_teacher: teacher_key=%s, batch_size=%d, "
        "group_sizes=%s",
        teacher_key,
        len(rollout_batch),
        {name: len(idxs) for name, idxs in groups.items()},
    )
    return groups


def reorder_logps_to_batch(
    rollout_batch_size: int,
    indices: list[int],
    logps: list[torch.Tensor] | None,
) -> list[torch.Tensor | None]:
    """Scatter per-teacher logp results back into the original batch order."""
    out: list[torch.Tensor | None] = [None] * rollout_batch_size
    if logps is None:
        return out
    if len(logps) != len(indices):
        raise ValueError(
            f"[MoPD] teacher logp count mismatch: got {len(logps)}, "
            f"expected {len(indices)}"
        )
    for slot, logp in zip(indices, logps):
        out[slot] = logp
    return out


def merge_per_teacher_logps(
    per_teacher_results: list[list[torch.Tensor | None]],
    rollout_batch_size: int,
) -> list[torch.Tensor]:
    """Combine the per-teacher scattered results into a dense list."""
    merged: list[torch.Tensor | None] = [None] * rollout_batch_size
    for results in per_teacher_results:
        for slot, val in enumerate(results):
            if val is None:
                continue
            if merged[slot] is not None:
                raise RuntimeError(
                    f"[MoPD] sample {slot} matched more than one teacher; "
                    "this should be impossible after group_trajectories_by_teacher."
                )
            merged[slot] = val

    missing = [i for i, v in enumerate(merged) if v is None]
    if missing:
        raise RuntimeError(
            f"[MoPD] {len(missing)} samples have no teacher assignment after "
            f"merge: {missing[:16]}{' ...' if len(missing) > 16 else ''}"
        )
    return [t for t in merged if t is not None]  # type: ignore[misc]


def compute_owning_teachers(
    per_teacher_results: list[list["torch.Tensor | None"]],
    teacher_names_ordered: list[str],
    rollout_batch_size: int,
) -> list[str]:
    """Return ``owning[slot] = teacher_name`` for every slot in the batch."""
    if not teacher_names_ordered:
        raise ValueError(
            "[MoPD] compute_owning_teachers: teacher_names_ordered is empty"
        )
    owning: list[str | None] = [None] * rollout_batch_size
    for t_idx, results in enumerate(per_teacher_results):
        if len(results) != rollout_batch_size:
            raise ValueError(
                f"[MoPD] compute_owning_teachers: per_teacher_results[{t_idx}] "
                f"has length {len(results)}, expected {rollout_batch_size}"
            )
        for slot, val in enumerate(results):
            if val is None:
                continue
            if owning[slot] is not None:
                raise RuntimeError(
                    f"[MoPD] compute_owning_teachers: slot {slot} owned by "
                    f"both {owning[slot]!r} and "
                    f"{teacher_names_ordered[t_idx]!r}"
                )
            owning[slot] = teacher_names_ordered[t_idx]
    missing = [i for i, v in enumerate(owning) if v is None]
    if missing:
        raise RuntimeError(
            f"[MoPD] compute_owning_teachers: {len(missing)} slot(s) have no "
            f"owner (indices: {missing[:16]}"
            f"{' ...' if len(missing) > 16 else ''})"
        )
    logger.info(
        "[MoPD] compute_owning_teachers: per-teacher slot counts = %s",
        {
            name: sum(1 for o in owning if o == name)
            for name in teacher_names_ordered
        },
    )
    return [o for o in owning if o is not None]  # type: ignore[misc]


def bucketize_indices_by_owner(
    owning_per_slot: list[str],
) -> dict[str, list[int]]:
    """Group adv-batch indices by their owning teacher."""
    buckets: dict[str, list[int]] = {}
    for slot, owner in enumerate(owning_per_slot):
        buckets.setdefault(owner, []).append(slot)
    logger.info(
        "[MoPD] bucketize_indices_by_owner: %d bucket(s), sizes=%s",
        len(buckets),
        {k: len(v) for k, v in buckets.items()},
    )
    return buckets


def teachers_share_uniform_weights(
    teachers_cfg: Mapping[str, "TeacherConfig"],
) -> bool:
    """Return True iff every configured teacher uses the same RL/distill weights."""
    if not teachers_cfg:
        return True
    iterator = iter(teachers_cfg.values())
    first = next(iterator)
    first_rl = float(getattr(first, "rl_loss_weight", 1.0))
    first_dl = float(getattr(first, "distill_loss_weight", 0.0))
    for t_cfg in iterator:
        rl = float(getattr(t_cfg, "rl_loss_weight", 1.0))
        dl = float(getattr(t_cfg, "distill_loss_weight", 0.0))
        if rl != first_rl or dl != first_dl:
            logger.info(
                "[MoPD] teachers_share_uniform_weights: weights differ — "
                "falling back to per-bucket update path."
            )
            return False
    logger.info(
        "[MoPD] teachers_share_uniform_weights: uniform across %d teacher(s) "
        "(rl=%s, distill=%s); using single ppo_update.",
        len(teachers_cfg),
        first_rl,
        first_dl,
    )
    return True
