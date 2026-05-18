# SPDX-License-Identifier: Apache-2.0

"""Combined GSM8K + ToRL RL dataset loader for Multi-teacher On-Policy
Distillation (MoPD).

This loader concatenates two *independent* text-only math datasets into a
single training stream where every row carries a distinct ``data_source``
routing key, so MoPD can dispatch GSM8K rows to one teacher and ToRL rows
to another teacher in the same step. It is the minimal way to exercise
"two physically separate datasets × two teachers" without introducing a
multimodal dataset (which would break the text-only Qwen3-0.6B student).

Why a dedicated combined loader
-------------------------------
The existing ``get_gsm8k_rl_dataset`` and ``get_torl_data_rl_dataset``
already produce HuggingFace ``Dataset`` objects with overlapping schema
(``messages``, ``answer``, ``data_source``). A naive ``concatenate_datasets``
on the caller side would still need to:

1. unify the per-row ``data_source`` values to two well-defined routing
   keys (ToRL's parquet carries fine-grained sub-source labels like
   "MATH"/"AIME"/"olympiad" which would explode the routing space);
2. drop columns that only one side has, otherwise ``concatenate_datasets``
   raises a feature-mismatch error;
3. respect ``max_length`` filtering uniformly.

This module centralises (1)–(3) behind a single entry point so the YAML
side stays declarative.

Schema after concat
-------------------
Every row has exactly:
  * ``messages``    — chat-template-ready ``list[dict]`` with one user turn
  * ``answer``      — ground-truth string (``"#### 42"`` for GSM8K,
                      ``"\\boxed{...}"`` for ToRL); both are accepted by
                      ``math_verify`` and consumed by ``gsm8k_reward_fn``.
  * ``data_source`` — routing key, exactly one of ``gsm8k_key`` /
                      ``torl_key`` (defaults: ``"openai/gsm8k"`` /
                      ``"torl_data"``).

Routing in MoPD
---------------
``PPOConfig.teacher_key`` is ``"data_source"``. Map each routing key
to a teacher via ``teachers.<name>.key`` in the YAML; the trainer's
``group_trajectories_by_teacher`` then dispatches GSM8K trajectories to
the GSM8K teacher and ToRL trajectories to the ToRL teacher within the
same global step.
"""

import logging
import os
import sys
from typing import Optional

from datasets import Dataset, concatenate_datasets

from .gsm8k import get_gsm8k_rl_dataset
from .torl_data import get_torl_data_rl_dataset

logger = logging.getLogger("areal.dataset.gsm8k_torl")


def _stderr_log(msg: str) -> None:
    """Mirror critical loader events to stderr so they survive even when the
    ``areal.dataset.gsm8k_torl`` logger is filtered out by a worker process
    (this is exactly what happened in mopd.log.3 — the ``logger.info`` lines
    never surfaced even though the loader clearly ran). Prefixed with PID so
    interleaving across rollout / trainer workers stays decipherable."""
    try:
        print(
            f"[gsm8k_torl][pid={os.getpid()}] {msg}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        # stderr write should never break the loader; swallow any encoding /
        # broken-pipe issues silently.
        pass


def _project_to_routing_schema(
    dataset: Dataset,
    routing_key: str,
) -> Dataset:
    """Strip every column that isn't part of the canonical routing schema
    and stamp a uniform ``data_source`` value on every row.

    ``concatenate_datasets`` requires identical features across operands;
    the cheapest way to guarantee that is to project both inputs onto the
    intersection schema before concatenation. We also overwrite
    ``data_source`` here so callers get a *single* routing key per
    sub-dataset regardless of any per-row labels carried in the source
    parquet (e.g. ToRL's "MATH"/"AIME").
    """
    keep = {"messages", "answer"}
    extra = [c for c in dataset.column_names if c not in keep]
    if extra:
        dataset = dataset.remove_columns(extra)

    def _stamp(_sample):
        return {"data_source": routing_key}

    # ``map`` adds the column uniformly; we do not pass ``with_indices``
    # because the routing key is a constant for this sub-dataset.
    return dataset.map(_stamp)


def _resize_dataset_to(
    dataset: Dataset,
    target_size: int,
    *,
    name: str,
    seed: int = 0,
) -> Dataset:
    """Up- or down-sample ``dataset`` to exactly ``target_size`` rows.

    * ``target_size <= len(dataset)`` → take a deterministic shuffled
      prefix (downsample).
    * ``target_size  > len(dataset)`` → repeat the dataset whole-times
      then append a deterministic shuffled prefix to fill the remainder
      (upsample). Whole repeats keep label balance; the residual is
      drawn after a fixed-seed shuffle so different ranks see the same
      rows.

    Both paths preserve features so the result still concats with peers
    that share the projected ``{messages, answer, data_source}`` schema.
    """
    n = len(dataset)
    if target_size == n:
        return dataset
    if n == 0:
        raise ValueError(
            f"_resize_dataset_to: cannot resize empty dataset {name!r}"
        )
    if target_size < n:
        # Downsample: deterministic shuffle then take prefix.
        shuffled = dataset.shuffle(seed=seed)
        out = shuffled.select(range(target_size))
        logger.info(
            "[gsm8k_torl] downsample %s: %d -> %d (seed=%d)",
            name,
            n,
            target_size,
            seed,
        )
        return out

    # Upsample: full repeats + deterministic residual.
    full_repeats = target_size // n
    residual = target_size - full_repeats * n
    parts = [dataset] * full_repeats
    if residual > 0:
        shuffled = dataset.shuffle(seed=seed)
        parts.append(shuffled.select(range(residual)))
    out = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    logger.info(
        "[gsm8k_torl] upsample %s: %d -> %d (full_repeats=%d, residual=%d, seed=%d)",
        name,
        n,
        target_size,
        full_repeats,
        residual,
        seed,
    )
    return out


def _compute_target_sizes(
    n_gsm8k: int,
    n_torl: int,
    gsm8k_share: float,
    mode: str,
) -> tuple[int, int]:
    """Compute (gsm8k_target, torl_target) so that
    ``gsm8k_target / (gsm8k_target + torl_target) == gsm8k_share``.

    Two modes:

    * ``"downsample"`` — keep the side that is *already* over-represented
      relative to the target share at its current size and shrink the
      other. Smaller pool, no row repetition; ideal when both sides have
      enough rows. Total rows = ``min(n_gsm8k / gsm8k_share, n_torl / (1 - gsm8k_share))``.
    * ``"upsample"`` — keep the side that is under-represented at its
      current size and grow the other by repeating rows. Larger pool,
      preserves all unique data; ideal when one side is very small.
      Total rows = ``max(n_gsm8k / gsm8k_share, n_torl / (1 - gsm8k_share))``.

    The function returns integer sizes; rounding is biased so the
    achieved ratio is within 1 row of the target.
    """
    if not (0.0 < gsm8k_share < 1.0):
        raise ValueError(
            f"_compute_target_sizes: gsm8k_share must be in (0, 1), got "
            f"{gsm8k_share!r}"
        )
    torl_share = 1.0 - gsm8k_share

    # The two candidate totals: shrink-to-min vs grow-to-max.
    # If we keep gsm8k at n_gsm8k, total = n_gsm8k / gsm8k_share.
    # If we keep torl  at n_torl,  total = n_torl  / torl_share.
    total_if_keep_gsm8k = n_gsm8k / gsm8k_share
    total_if_keep_torl = n_torl / torl_share

    if mode == "downsample":
        total = min(total_if_keep_gsm8k, total_if_keep_torl)
    elif mode == "upsample":
        total = max(total_if_keep_gsm8k, total_if_keep_torl)
    else:
        raise ValueError(
            f"_compute_target_sizes: mode must be 'downsample' or 'upsample', "
            f"got {mode!r}"
        )

    gsm8k_target = max(1, int(round(total * gsm8k_share)))
    torl_target = max(1, int(round(total * torl_share)))
    return gsm8k_target, torl_target


def get_gsm8k_torl_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    *,
    gsm8k_path: str = "openai/gsm8k",
    torl_path: str = "/tmp/areal/torl_data/train.parquet",
    gsm8k_split: Optional[str] = None,
    torl_split: Optional[str] = None,
    gsm8k_key: str = "openai/gsm8k",
    torl_key: str = "torl_data",
    gsm8k_share: Optional[float] = None,
    ratio_mode: str = "downsample",
    resize_seed: int = 0,
):
    """Load GSM8K + ToRL as a single concatenated RL dataset for MoPD.

    Parameters
    ----------
    path
        Ignored — accepted only because ``_get_custom_dataset`` always
        forwards ``dataset_config.path``. Use the dispatcher key
        ``"gsm8k_torl"`` in the YAML to select this loader. Kept for API
        symmetry with the other ``get_*_rl_dataset`` functions.
    split
        Default split name. When ``gsm8k_split``/``torl_split`` are not
        supplied, both sub-datasets fall back to this value (HuggingFace
        ``load_dataset`` semantics for GSM8K; ToRL's parquet is loaded as
        a single ``"train"`` regardless and ignores this arg upstream).
    tokenizer
        Tokenizer used for length-filtering after concat.
    max_length
        If set, drop rows whose tokenised user content exceeds this many
        tokens. Filtering happens *after* concat so both sides use the
        same threshold and the same tokenizer.
    gsm8k_path, torl_path
        Override the underlying dataset paths if needed.
    gsm8k_split, torl_split
        Per-side split overrides; default to ``split``.
    gsm8k_key, torl_key
        Routing keys stamped into the ``data_source`` column. These must
        match the ``key`` of the corresponding ``teachers.<name>`` entry
        in the YAML.
    gsm8k_share
        Target proportion of GSM8K rows in the concatenated pool, in
        ``(0, 1)``. When ``None`` (default), no resizing happens — the
        concat ratio is whatever the raw datasets give (≈ 20:80 for the
        default GSM8K-train + ToRL-train pair). When set (e.g. ``0.9``),
        one side is shuffled-and-resized so the final pool matches the
        requested share to within one row. Because
        ``DistributedSampler(shuffle=True)`` draws every batch uniformly
        from this pool, the per-step expected GSM8K:ToRL ratio equals
        ``gsm8k_share : (1 - gsm8k_share)``.
    ratio_mode
        How to achieve ``gsm8k_share``:

        * ``"downsample"`` (default) — shrink the over-represented side.
          Smaller training pool, no row repetition. Recommended when the
          larger side has plenty of rows to spare. For
          ``gsm8k_share=0.9`` with default sizes this keeps GSM8K at
          ~7473 rows and shrinks ToRL from ~28k to ~830 rows.
        * ``"upsample"`` — grow the under-represented side via whole
          repeats + a deterministic shuffled residual. Larger pool,
          preserves every unique row.
    resize_seed
        Seed used for the deterministic shuffle that drives down/up
        sampling. Fixed default (``0``) so every rank sees the same
        resized pool.

    Returns
    -------
    datasets.Dataset
        Concatenated dataset with columns ``{messages, answer,
        data_source}`` and rows ordered as ``[gsm8k_rows..., torl_rows...]``.
        Downstream ``DataLoader(shuffle=True)`` ensures inter-source
        interleaving within each batch.
    """
    del path  # see docstring — accepted for dispatcher symmetry only
    # Unconditional entry log — surfaces the loader call site + the
    # exact kwargs the YAML / CLI override passed in. If ``gsm8k_share``
    # is missing here it is also missing in the resulting concat pool,
    # which then routes ~80% to ToRL by default.
    logger.info(
        "[gsm8k_torl] get_gsm8k_torl_rl_dataset called: split=%r "
        "gsm8k_path=%r torl_path=%r gsm8k_split=%r torl_split=%r "
        "gsm8k_key=%r torl_key=%r gsm8k_share=%r ratio_mode=%r "
        "resize_seed=%r max_length=%r",
        split,
        gsm8k_path,
        torl_path,
        gsm8k_split,
        torl_split,
        gsm8k_key,
        torl_key,
        gsm8k_share,
        ratio_mode,
        resize_seed,
        max_length,
    )
    # Mirror entry kwargs to stderr — worker stdout in mopd.log.3 never
    # surfaced any ``[gsm8k_torl]`` line, which made it impossible to tell
    # whether ``gsm8k_share`` actually reached the loader. stderr survives
    # most logger configurations.
    _stderr_log(
        f"ENTRY split={split!r} gsm8k_path={gsm8k_path!r} "
        f"torl_path={torl_path!r} gsm8k_split={gsm8k_split!r} "
        f"torl_split={torl_split!r} gsm8k_key={gsm8k_key!r} "
        f"torl_key={torl_key!r} gsm8k_share={gsm8k_share!r} "
        f"ratio_mode={ratio_mode!r} resize_seed={resize_seed!r} "
        f"max_length={max_length!r}"
    )
    if gsm8k_key == torl_key:
        raise ValueError(
            "get_gsm8k_torl_rl_dataset: 'gsm8k_key' and 'torl_key' must "
            f"differ for MoPD routing to be meaningful, got {gsm8k_key!r}"
        )

    gsm8k_ds = get_gsm8k_rl_dataset(
        path=gsm8k_path,
        split=gsm8k_split if gsm8k_split is not None else split,
        tokenizer=tokenizer,
        # length filtering applied uniformly *after* concat so both
        # sub-datasets see the same threshold under the same tokenizer.
        max_length=None,
        # Stamp the routing key directly via the underlying loader — it
        # already supports a uniform ``data_source`` override.
        data_source=gsm8k_key,
    )

    torl_ds = get_torl_data_rl_dataset(
        path=torl_path,
        split=torl_split if torl_split is not None else split,
        tokenizer=tokenizer,
        max_length=None,
        data_source=torl_key,
    )

    # Project both onto the intersection schema {messages, answer} and
    # stamp ``data_source`` deterministically. ``_project_to_routing_schema``
    # must run on *both* operands so feature dicts match exactly.
    gsm8k_ds = _project_to_routing_schema(gsm8k_ds, routing_key=gsm8k_key)
    torl_ds = _project_to_routing_schema(torl_ds, routing_key=torl_key)

    # Sanity: features must agree column-by-column for concat to succeed.
    if set(gsm8k_ds.column_names) != set(torl_ds.column_names):
        raise RuntimeError(
            "get_gsm8k_torl_rl_dataset: schema mismatch after projection — "
            f"gsm8k columns {gsm8k_ds.column_names} vs torl columns "
            f"{torl_ds.column_names}. This indicates the upstream loaders "
            "drifted; please re-check get_gsm8k_rl_dataset / "
            "get_torl_data_rl_dataset return shapes."
        )

    # Optional ratio enforcement. When ``gsm8k_share`` is None we keep
    # both sub-datasets at their natural sizes (legacy behaviour). When
    # set, resize one (or both) so the concat pool achieves the
    # requested GSM8K:ToRL split. ``DistributedSampler(shuffle=True)``
    # then draws uniformly from the resized pool, so the per-step
    # *expected* ratio equals ``gsm8k_share : (1 - gsm8k_share)``.
    if gsm8k_share is None:
        # Make the no-share path visible — if a user expects 90/10 and
        # forgot to pass the kwarg, the routing distribution will look
        # wrong (ToRL-dominated by default because ToRL has ~28k rows
        # vs GSM8K ~7.5k); we want this discoverable in the log
        # without requiring DEBUG verbosity.
        logger.info(
            "[gsm8k_torl] gsm8k_share=None — keeping natural sizes "
            "(gsm8k=%d, torl=%d, total=%d, natural gsm8k_share=%.4f). "
            "If you intended a specific ratio, pass "
            "++train_dataset.dataset_kwargs.gsm8k_share=0.9 (or similar).",
            len(gsm8k_ds),
            len(torl_ds),
            len(gsm8k_ds) + len(torl_ds),
            len(gsm8k_ds) / max(1, len(gsm8k_ds) + len(torl_ds)),
        )
        _stderr_log(
            f"NO-SHARE-MODE gsm8k={len(gsm8k_ds)} torl={len(torl_ds)} "
            f"total={len(gsm8k_ds) + len(torl_ds)} "
            f"natural_gsm8k_share="
            f"{len(gsm8k_ds) / max(1, len(gsm8k_ds) + len(torl_ds)):.4f}"
        )
    if gsm8k_share is not None:
        n_g, n_t = len(gsm8k_ds), len(torl_ds)
        gsm8k_target, torl_target = _compute_target_sizes(
            n_gsm8k=n_g,
            n_torl=n_t,
            gsm8k_share=float(gsm8k_share),
            mode=ratio_mode,
        )
        gsm8k_ds = _resize_dataset_to(
            gsm8k_ds, gsm8k_target, name="gsm8k", seed=resize_seed
        )
        torl_ds = _resize_dataset_to(
            torl_ds, torl_target, name="torl", seed=resize_seed
        )
        achieved_share = gsm8k_target / float(gsm8k_target + torl_target)
        logger.info(
            "[gsm8k_torl] ratio enforcement: target gsm8k_share=%.4f, "
            "mode=%s, sizes (gsm8k, torl) before=(%d, %d) after=(%d, %d), "
            "achieved gsm8k_share=%.4f",
            gsm8k_share,
            ratio_mode,
            n_g,
            n_t,
            gsm8k_target,
            torl_target,
            achieved_share,
        )
        _stderr_log(
            f"RATIO-ENFORCED target_gsm8k_share={gsm8k_share:.4f} "
            f"mode={ratio_mode} before=(gsm8k={n_g}, torl={n_t}) "
            f"after=(gsm8k={gsm8k_target}, torl={torl_target}) "
            f"achieved_gsm8k_share={achieved_share:.4f}"
        )

    combined = concatenate_datasets([gsm8k_ds, torl_ds])

    if max_length is not None:

        def _filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        combined = combined.filter(_filter_length)

    # Final-state diagnostic — counts of each routing key in the *post-filter*
    # pool, mirrored to stderr. This is the ground truth that
    # ``DistributedSampler(shuffle=True)`` ultimately samples from. If the
    # observed per-step gsm8k:torl ratio deviates, the cause is upstream of
    # the sampler (e.g. config not applied, max_length filter biasing one
    # side, or different DataLoader collators).
    try:
        ds_field = combined["data_source"]
        post_counts: dict[str, int] = {}
        for v in ds_field:
            post_counts[v] = post_counts.get(v, 0) + 1
        total = sum(post_counts.values()) or 1
        ratio_str = ", ".join(
            f"{k!r}: n={c} share={c / total:.4f}"
            for k, c in sorted(post_counts.items())
        )
        logger.info(
            "[gsm8k_torl] final pool after concat+filter: total=%d %s",
            total,
            ratio_str,
        )
        _stderr_log(
            f"FINAL-POOL total={total} " + ratio_str
        )
    except Exception as e:
        # Never break loading; the diagnostic is best-effort.
        _stderr_log(f"FINAL-POOL diagnostic FAILED: {e!r}")

    return combined
