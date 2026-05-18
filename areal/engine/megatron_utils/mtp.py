# SPDX-License-Identifier: Apache-2.0
"""MTP (Multi-Token Prediction) helpers for online MTP training.

Design goals
------------
1. **Optional** — every helper here only activates when the user explicitly
   sets ``MegatronEngineConfig.enable_mtp_training=True``. When the flag is off
   no helper is invoked, preserving original AReaL behaviour byte-for-byte.
2. **Non-invasive** — we never rewrite or monkey-patch upstream Megatron
   files. Instead, we walk the GPTModel module tree at runtime and call
   ``model.mtp(...)`` directly on the last pipeline stage.
3. **MTP-only gradients** — the MTP CE loss gradients must NOT flow back into
   the shared embedding / output_layer. We enforce this by detaching
   ``shared_embedding_or_output_weight`` before feeding it to ``model.mtp``.
4. **Instrumented** — every newly-added code path emits explicit ``[AReaL][MTP]``
   prefixed prints so production failures can be located with grep.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from areal.utils import logging, stats_tracker

logger = logging.getLogger("MTP")


# ---------------------------------------------------------------------------
# Module-tree helpers
# ---------------------------------------------------------------------------

def _unwrap_to_gpt_model(model: torch.nn.Module) -> torch.nn.Module:
    """Walk ``.module`` / ``.language_model`` until we hit something that has
    an ``mtp`` attribute (i.e. the inner GPTModel constructed by mbridge with
    ``num_nextn_predict_layers > 0``).

    Returns the original model unchanged if no MTP submodule is found.
    """
    cur = model
    # Try unwrapping DDP / Float16Module / mbridge wrapper
    for _ in range(8):
        if hasattr(cur, "mtp"):
            return cur
        if hasattr(cur, "module"):
            cur = cur.module
            continue
        if hasattr(cur, "language_model"):
            cur = cur.language_model
            continue
        break
    return cur


def model_has_mtp(model: torch.nn.Module) -> bool:
    """Quick sanity check used by the engine to gate MTP loss computation."""
    inner = _unwrap_to_gpt_model(model)
    return getattr(inner, "mtp", None) is not None and getattr(
        inner, "mtp_process", False
    )


# ---------------------------------------------------------------------------
# Inner MTP path runtime disabler
# ---------------------------------------------------------------------------


class disable_inner_mtp_path:
    """Context manager that temporarily disables Megatron-Core GPTModel's
    *inner* MTP forward path.

    Why this exists
    ---------------
    Megatron-Core's ``GPTModel._postprocess()`` runs an MTP forward + MTP CE
    loss block whenever ``self.mtp_process is True`` AND ``self.config.
    mtp_num_layers is not None``::

        if self.mtp_process:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                hidden_states=hidden_states,
                ...,
            )
        if self.config.mtp_num_layers is not None:
            mtp_labels = labels.clone()  # crashes when labels is None
            ...

    That path requires ``labels`` and ``loss_mask`` to be passed via
    ``extra_block_kwargs``.  AReaL never passes them in either inference or
    training (it computes MTP loss EXTERNALLY via
    :class:`MTPHiddenStateCapturer` + :func:`compute_mtp_loss`), so the inner
    path crashes with errors like::

        AttributeError: 'NoneType' object has no attribute 'clone'
        TypeError: ... missing 1 required positional argument: 'labels'

    The crash propagates up through ``packed_context_parallel_forward``
    (areal/engine/megatron_utils/packed_context_parallel.py:372) which
    masks the original exception by re-raising as ``RuntimeError(
    "Error occurred in packed context parallel forward pass on model
    {model} with input_ids shape ... packed_seq_params=...")``.

    AReaL does not need the inner MTP path at all -- the external capturer
    + ``compute_mtp_loss`` wires MTP into the actor loss with full gradient
    isolation.  So we *temporarily* turn the inner path OFF for the
    duration of each ``model(...)`` call, then restore the original
    attributes in ``__exit__`` so the saved/checkpointed module remains
    intact for serialization.

    Why this is safe
    ----------------
    1. Backbone forward is unaffected -- ``mtp`` is only consulted inside
       ``_postprocess`` AFTER the backbone has already produced
       ``hidden_states``.  The MTPHiddenStateCapturer hook on
       ``decoder.final_layernorm`` still fires.
    2. ``mtp_process`` is recomputed at engine init based on PP rank, so
       toggling it at runtime is reversible.
    3. ``config.mtp_num_layers`` is only read inside ``_postprocess``
       during the same forward call, so restoring it in ``__exit__``
       keeps the model identical pre/post-call.
    4. Idempotent + nest-safe via the ``_areal_inner_mtp_disabled`` sentinel
       on the inner model.  A nested ``with`` block is a no-op.

    Where to use
    ------------
    Wrap every ``packed_context_parallel_forward(model, ...)`` call inside
    ``MegatronEngine.forward_backward_batch::forward_step`` -- BOTH the
    inference path (forward_only=True; ppo_actor logprob collection) AND
    the training path (forward_only=False; ppo_update).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._inner: Optional[torch.nn.Module] = None
        self._saved_mtp: Any = None
        self._saved_mtp_process: Any = False
        self._saved_mtp_num_layers: Any = None
        self._had_mtp_attr: bool = False
        self._had_mtp_process_attr: bool = False
        self._had_mtp_num_layers_attr: bool = False
        self._activated: bool = False

    def __enter__(self) -> "disable_inner_mtp_path":
        try:
            inner = _unwrap_to_gpt_model(self._model)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "[AReaL][MTP][innerdisable] _unwrap_to_gpt_model failed: %s; "
                "skipping inner-MTP disable.",
                exc,
            )
            return self
        self._inner = inner

        # Nest-safe: another context already disabled the inner path.
        if getattr(inner, "_areal_inner_mtp_disabled", False):
            logger.debug(
                "[AReaL][MTP][innerdisable] inner MTP path already disabled "
                "by an outer context; this entry is a no-op."
            )
            return self

        self._had_mtp_attr = hasattr(inner, "mtp")
        self._saved_mtp = getattr(inner, "mtp", None)
        self._had_mtp_process_attr = hasattr(inner, "mtp_process")
        self._saved_mtp_process = getattr(inner, "mtp_process", False)

        cfg = getattr(inner, "config", None)
        if cfg is not None:
            self._had_mtp_num_layers_attr = hasattr(cfg, "mtp_num_layers")
            self._saved_mtp_num_layers = getattr(cfg, "mtp_num_layers", None)

        # Only act if the inner path is actually live; otherwise this is a
        # pure no-op and we should not pollute logs nor risk later restore.
        live = (
            self._saved_mtp is not None
            or bool(self._saved_mtp_process)
            or self._saved_mtp_num_layers is not None
        )
        if not live:
            logger.debug(
                "[AReaL][MTP][innerdisable] inner MTP path already inactive; "
                "no toggle needed."
            )
            return self

        try:
            inner.mtp = None
            inner.mtp_process = False
            if cfg is not None and self._had_mtp_num_layers_attr:
                cfg.mtp_num_layers = None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[AReaL][MTP][innerdisable] failed to disable inner MTP "
                "attributes (%s); proceeding without disable.",
                exc,
            )
            return self

        inner._areal_inner_mtp_disabled = True
        self._activated = True
        logger.debug(
            "[AReaL][MTP][innerdisable] disabled inner MTP path "
            "(saved mtp=%s, mtp_process=%s, mtp_num_layers=%s).",
            type(self._saved_mtp).__name__ if self._saved_mtp is not None else None,
            self._saved_mtp_process,
            self._saved_mtp_num_layers,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        inner = self._inner
        if inner is None or not self._activated:
            return
        try:
            if self._had_mtp_attr:
                inner.mtp = self._saved_mtp
            else:
                try:
                    delattr(inner, "mtp")
                except AttributeError:
                    pass
            if self._had_mtp_process_attr:
                inner.mtp_process = self._saved_mtp_process
            else:
                try:
                    delattr(inner, "mtp_process")
                except AttributeError:
                    pass
            cfg = getattr(inner, "config", None)
            if cfg is not None and self._had_mtp_num_layers_attr:
                cfg.mtp_num_layers = self._saved_mtp_num_layers
        finally:
            try:
                delattr(inner, "_areal_inner_mtp_disabled")
            except AttributeError:
                pass
            logger.debug(
                "[AReaL][MTP][innerdisable] restored inner MTP path."
            )


def _shared_output_weight_detached(inner_gpt_model: torch.nn.Module) -> torch.Tensor:
    """Return ``inner_gpt_model.shared_embedding_or_output_weight().detach()``.

    Detaching is the key invariant — it prevents the MTP CE loss from polluting
    the gradients of the main embedding / output_layer parameters.
    """
    if hasattr(inner_gpt_model, "shared_embedding_or_output_weight"):
        weight = inner_gpt_model.shared_embedding_or_output_weight()
    else:  # pragma: no cover - defensive
        # Fallback: directly use output_layer.weight
        weight = inner_gpt_model.output_layer.weight
    return weight.detach()


# ---------------------------------------------------------------------------
# MTP forward
# ---------------------------------------------------------------------------

def compute_mtp_loss(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    packed_seq_params: Any = None,
) -> Optional[torch.Tensor]:
    """[v8] Deprecated -- always returns None.

    History
    -------
    * v3-v7: This function attempted to forward Megatron's
      ``MultiTokenPredictionBlock`` externally and compute MTP CE loss
      ourselves, then add the scaled value to AReaL's actor loss.
    * v8 (this revision): the external path is unreliable across
      megatron-core versions because:
        - ``MultiTokenPredictionBlock.forward`` upstream signature does NOT
          accept ``output_weight`` / ``labels`` / ``loss_mask`` (those are
          consumed by ``process_mtp_loss``, not by the block).  The kwarg
          call therefore raises ``TypeError: ... got an unexpected keyword
          argument 'output_weight'``.
        - The positional fallback then mis-aligns args (passing a tensor
          as ``position_ids``, etc.) and ultimately triggers
          ``'NoneType' object is not callable`` deep inside the block.
        - Reproducing ``process_mtp_loss``'s segment chunking, label
          rolling under context parallelism, and ``MTPLossAutoScaler``
          gradient injection externally is brittle and version-sensitive.

    v8 routes MTP loss through Megatron's *upstream* ``_postprocess`` ->
    ``process_mtp_loss`` path by:
      1. Passing ``labels`` and ``loss_mask`` to ``model.forward`` via
         :class:`MTPAwareForward` (which patches
         ``compute_language_model_loss`` to return ``logits`` so AReaL
         still receives logits, not CE loss).
      2. ``MTPLossAutoScaler`` then injects the MTP gradient into the
         backbone backward pass automatically.

    The external capturer + this helper are now no-ops; we keep the symbol
    for backward source-compatibility with any caller that imports
    ``compute_mtp_loss`` and expects a callable returning ``Optional[Tensor]``.
    """
    if not getattr(compute_mtp_loss, "_v8_deprecation_logged", False):
        try:
            logger.info(
                "[AReaL][MTP] compute_mtp_loss is deprecated as of v8; MTP "
                "loss now flows through Megatron's upstream "
                "_postprocess->process_mtp_loss path via MTPAwareForward."
            )
        except Exception:
            pass
        compute_mtp_loss._v8_deprecation_logged = True
    return None


# ---------------------------------------------------------------------------
# MTP-aware forward orchestrator (v8)
# ---------------------------------------------------------------------------


class MTPAwareForward:
    """v8 context manager that prepares ``model.forward`` for AReaL +
    Megatron-Core MTP without crashing.

    Three modes
    -----------
    1. **No MTP on the model** -- pure no-op; ``extra_model_kwargs`` is ``{}``.
    2. **Inference (forward_only=True) + has MTP** -- temporarily nulls
       ``inner.mtp`` / ``inner.mtp_process`` / ``inner.config.mtp_num_layers``
       (the v7 ``disable_inner_mtp_path`` behavior) so that
       ``_postprocess`` short-circuits and returns logits without trying
       to call ``self.mtp(...)`` (which crashes on ``labels.clone()`` when
       AReaL's logprob-collection path doesn't pass labels).
    3. **Training (forward_only=False) + has MTP** -- enables Megatron's
       upstream MTP loss path:
         a. Provides ``extra_model_kwargs = {"labels": input_ids,
            "loss_mask": loss_mask_or_ones}`` so ``_postprocess``'s
            ``process_mtp_loss`` branch (line 651 in megatron-core
            ``gpt_model.py``) computes the per-MTP-layer CE loss and
            attaches the gradient to the backbone hidden states via
            ``MTPLossAutoScaler``.
         b. Patches ``inner.compute_language_model_loss(labels, logits)``
            so that the FIRST ``mtp_num_layers`` calls (issued from
            ``process_mtp_loss``'s per-layer loop) return real CE loss
            (required for ``mtp_loss = loss_mask * mtp_loss``), and the
            ``mtp_num_layers + 1``-th call (issued from
            ``_postprocess``'s final ``loss = compute_language_model_loss(
            labels, logits)`` line) returns ``logits.transpose(0, 1).contiguous()``
            so AReaL's RL loss pipeline keeps receiving logits unchanged.
            The counter is reset at every ``_postprocess`` entry by an
            additional ``_postprocess`` wrapper.
         c. Restores both on exit.

    Usage::

        with MTPAwareForward(model, mb_input.padded_mb, forward_only) as cm:
            output = packed_context_parallel_forward(
                model,
                mb_input.padded_mb,
                gather_cp_output=...,
                is_vision_model=...,
                extra_model_kwargs=cm.extra_model_kwargs,
            )

    Why this is correct (v12)
    -------------------------
    * Instead of patching ``compute_language_model_loss`` (which is a
      fragile state-machine: caller must guess how many times CLM is
      called by ``process_mtp_loss`` + ``_postprocess``), v12 REPLACES
      ``GPTModel._postprocess`` outright with a method that mirrors
      megatron-core 0.17.0 ``gpt_model.py::_postprocess`` line-for-line
      up to the MTP block, then inlines the MTP loss loop using the
      *un-patched* ``compute_language_model_loss`` for real per-token
      CE, and finally returns ``logits.transpose(0, 1).contiguous()`` --
      never reaching the upstream's final ``loss = compute_language_
      model_loss(labels, logits)`` line.
    * This matches the strategy used by AReaL PR #1176 (TaoZex/spec_v1)
      and slime under THUDM Megatron, both of which install a full
      ``_postprocess`` replacement rather than juggling CLM patches.
    * ``MTPLossAutoScaler.apply(hidden_states, mtp_loss)`` is invoked
      inline, so the MTP CE gradient is injected into the backbone
      backward pass exactly as upstream's ``process_mtp_loss`` would
      have done.  Forward returns ``hidden_states`` unchanged.
    * ``loss_mask`` is optional; when ``None``, the inline loop uses
      ``torch.ones_like(mtp_labels)`` (mirroring upstream).
    * Non-training paths (``self_model.training is False``, inference
      contexts, speculative decoding) DEFER to the original
      ``_postprocess`` so we don't re-implement upstream caching /
      inference behaviour.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mb_input: dict,
        forward_only: bool,
    ) -> None:
        self._model = model
        self._mb_input = mb_input
        self._forward_only = forward_only
        self._inner: Optional[torch.nn.Module] = None
        self._has_mtp: bool = False
        self._activated: bool = False

        # Inference-mode disable-state
        self._saved_mtp: Any = None
        self._saved_mtp_process: Any = False
        self._saved_mtp_num_layers: Any = None
        self._had_mtp_attr: bool = False
        self._had_mtp_process_attr: bool = False
        self._had_mtp_num_layers_attr: bool = False

        # Training-mode patch state
        self._patched_compute_lm_loss: bool = False
        self._orig_compute_lm_loss: Any = None
        # v11: also wrap _postprocess so the CLM call counter resets on every
        # invocation (process_mtp_loss calls CLM N times, then _postprocess
        # calls it once at the end -- the N+1-th call must return logits, the
        # first N must return real CE loss for ``mtp_loss = loss_mask * mtp_loss``).
        self._patched_postprocess: bool = False
        self._orig_postprocess: Any = None
        # Per-_postprocess-invocation CLM call counter, mutated by the wrapper
        # and read by the patched CLM. Lives on the CM so multiple in-flight
        # forwards (different ranks / micro-batches) cannot collide because
        # each rank has its own MTPAwareForward instance.
        self._clm_call_counter: list[int] = [0]

        # extra_model_kwargs for the model() call (training mode only)
        self.extra_model_kwargs: dict = {}

    def __enter__(self) -> "MTPAwareForward":
        try:
            inner = _unwrap_to_gpt_model(self._model)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "[AReaL][MTP][v8] _unwrap_to_gpt_model failed: %s; "
                "running plain forward.",
                exc,
            )
            return self
        self._inner = inner

        self._has_mtp = (
            getattr(inner, "mtp", None) is not None
            or bool(getattr(inner, "mtp_process", False))
            or getattr(getattr(inner, "config", None), "mtp_num_layers", None)
            is not None
        )

        if not self._has_mtp:
            return self

        if self._forward_only:
            self._enter_inference_mode(inner)
        else:
            self._enter_training_mode(inner)
        return self

    def _enter_inference_mode(self, inner: torch.nn.Module) -> None:
        # Mirror v7 disable_inner_mtp_path semantics.
        if getattr(inner, "_areal_mtpaware_disabled", False):
            logger.debug(
                "[AReaL][MTP][v8] inner MTP already disabled by outer ctx."
            )
            return
        self._had_mtp_attr = hasattr(inner, "mtp")
        self._saved_mtp = getattr(inner, "mtp", None)
        self._had_mtp_process_attr = hasattr(inner, "mtp_process")
        self._saved_mtp_process = getattr(inner, "mtp_process", False)
        cfg = getattr(inner, "config", None)
        if cfg is not None:
            self._had_mtp_num_layers_attr = hasattr(cfg, "mtp_num_layers")
            self._saved_mtp_num_layers = getattr(cfg, "mtp_num_layers", None)
        try:
            inner.mtp = None
            inner.mtp_process = False
            if cfg is not None and self._had_mtp_num_layers_attr:
                cfg.mtp_num_layers = None
            inner._areal_mtpaware_disabled = True
            self._activated = True
            logger.debug(
                "[AReaL][MTP][v8] inference: disabled inner MTP path."
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[AReaL][MTP][v8] inference: failed to disable inner MTP "
                "(%s); proceeding without disable.",
                exc,
            )

    def _enter_training_mode(self, inner: torch.nn.Module) -> None:
        # ====================================================================
        # v12: Replace ``_postprocess`` outright instead of patching CLM.
        # ====================================================================
        # Why v11 (counter-based CLM patch) is fragile:
        #   * It relies on ``process_mtp_loss`` calling
        #     ``compute_language_model_loss`` exactly ``mtp_num_layers`` times
        #     before ``_postprocess``'s final call.  Any megatron-core code
        #     path or future revision that adds / skips a CLM call (e.g.
        #     speculative decoding, partial-PP-stage shortcuts, custom logits
        #     processors) breaks the contract silently.
        #
        # v12 takes the same approach as PR #1176 and slime under THUDM
        # Megatron: replace ``GPTModel._postprocess`` entirely.  The
        # replacement mirrors megatron-core 0.17.0 ``gpt_model.py::_postprocess``
        # line-for-line up to the MTP block, runs the MTP loss loop INLINE
        # (calling the *unmodified* ``compute_language_model_loss`` once per
        # MTP layer for real per-token CE loss), and finally returns
        # ``logits.transpose(0, 1).contiguous()`` -- exactly what AReaL's RL
        # pipeline expects.  No CLM patch is required.
        #
        # Build labels / loss_mask from the packed micro-batch.
        # ``packed_context_parallel_forward`` reshapes them in lock-step with
        # ``input_ids`` (v10 fix), so by the time the model receives them
        # they are in the post-CP-split, post-pack form that
        # ``packed_seq_params`` describes.
        labels = self._mb_input.get("input_ids", None)
        if labels is None:
            logger.warning(
                "[AReaL][MTP] training: no input_ids in mb_input; "
                "skipping MTP loss injection."
            )
            return
        loss_mask = self._mb_input.get("loss_mask", None)

        self.extra_model_kwargs = {"labels": labels}
        if loss_mask is not None:
            # Cast to float; the MTP loop multiplies (loss_mask * mtp_loss).
            self.extra_model_kwargs["loss_mask"] = loss_mask.float()

        # Replace _postprocess.  We keep a reference to the original bound
        # method so __exit__ can restore it.
        if not hasattr(inner, "_postprocess"):
            logger.warning(
                "[AReaL][MTP] training: inner model has no "
                "_postprocess; cannot install MTP-aware replacement."
            )
            return
        if getattr(
            inner._postprocess,
            "_areal_mtpaware_v12_replacement",
            False,
        ):
            self._activated = True
            return

        # Capture the original UNBOUND function so we can fall back to it on
        # paths we don't handle (inference / spec-decode).  ``inner._postprocess``
        # is a bound method; ``__func__`` gives us the underlying function.
        try:
            _orig_unbound = inner._postprocess.__func__
        except AttributeError:
            # Already a plain function (some test mocks); use it directly.
            _orig_unbound = inner._postprocess
        self._orig_postprocess = inner._postprocess  # bound, for restore

        import types as _types_mod
        from megatron.core.transformer.multi_token_prediction import (
            MTPLossAutoScaler,
            roll_tensor,
        )

        # Snapshot mtp_num_layers at patch time so the replacement is robust
        # against config mutation mid-step.
        _cfg = getattr(inner, "config", None)
        _mtp_num_layers = int(getattr(_cfg, "mtp_num_layers", 0) or 0)
        _mtp_loss_scaling_factor = float(
            getattr(_cfg, "mtp_loss_scaling_factor", 0.1) or 0.1
        )
        _calculate_per_token_loss = bool(
            getattr(_cfg, "calculate_per_token_loss", False)
        )

        # ----------------------------------------------------------------
        # v19: one-time MTP weight fingerprint (per-rank, per-process).
        #
        # Why: spec_v1.log.15 shows step-0 ``mtp/layer0/loss_avg ≈ 10.91``
        # which is suspiciously close to ``ln(vocab_size)`` and could
        # indicate a partially uninitialised MTP head (i.e. HF -> Megatron
        # load skipping eh_proj / input_proj / *layernorm). To confirm or
        # falsify this without intrusive prints, emit a one-shot
        # fingerprint of the four MTP-specific submodule weights: their
        # L2 norms, mean, and (for eh_proj) the absolute mean of the
        # FIRST-half vs SECOND-half columns. A correctly-loaded pretrained
        # MiMo MTP head produces fingerprints close to those of the base
        # decoder layer-0 (norms in the same order of magnitude); a
        # randomly-initialised head fingerprints near zero (norms several
        # orders of magnitude smaller). The user can grep
        # ``[AReaL][MTP][fingerprint]`` once at startup to verify.
        #
        # Aligned with slime: slime's ``_weight_to_mcore_format`` swaps
        # eh_proj column halves on HF -> Megatron load; checking the
        # post-swap fingerprint here is exactly what surfaces a load
        # failure if it ever occurs again.
        # ----------------------------------------------------------------
        try:
            if not getattr(inner, "_areal_mtp_fingerprint_logged", False):
                with torch.no_grad():
                    _fp_lines = []
                    _mtp_module = getattr(inner, "mtp", None)
                    _layers = (
                        list(getattr(_mtp_module, "layers", []))
                        if _mtp_module is not None
                        else []
                    )
                    for _li, _layer in enumerate(_layers[:2]):
                        for _attr in (
                            "enorm",
                            "hnorm",
                            "eh_proj",
                            "final_layernorm",
                        ):
                            _sub = getattr(_layer, _attr, None)
                            _w = getattr(_sub, "weight", None)
                            if _w is None:
                                continue
                            _wd = _w.detach().float()
                            _norm = float(_wd.norm().item())
                            _mean = float(_wd.abs().mean().item())
                            _extra = ""
                            if _attr == "eh_proj" and _wd.dim() == 2 \
                                    and _wd.shape[1] % 2 == 0:
                                _h1, _h2 = _wd.chunk(2, dim=1)
                                _extra = (
                                    f" half1_abs_mean={float(_h1.abs().mean()):.4e}"
                                    f" half2_abs_mean={float(_h2.abs().mean()):.4e}"
                                )
                            _fp_lines.append(
                                f"layer{_li}.{_attr}: shape={tuple(_wd.shape)} "
                                f"l2={_norm:.4e} abs_mean={_mean:.4e}{_extra}"
                            )
                    if _fp_lines:
                        logger.info(
                            "[AReaL][MTP][fingerprint] one-time MTP weight "
                            "fingerprint at training entry: %s",
                            " | ".join(_fp_lines),
                        )
                inner._areal_mtp_fingerprint_logged = True
        except Exception as _fp_exc:  # pragma: no cover
            logger.debug(
                "[AReaL][MTP][fingerprint] emission failed: %r", _fp_exc
            )

        def _patched_postprocess(
            self_model,
            hidden_states,
            input_ids,
            position_ids,
            labels,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            mtp_in_postprocess=None,
            loss_mask=None,
            decoder_input=None,
            attention_mask=None,
            inference_params=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            runtime_gather_output=None,
            extra_block_kwargs=None,
            inference_context=None,
            is_spec_decode=None,
        ):
            """v12 replacement for GPTModel._postprocess.

            Mirrors megatron-core 0.17.0 gpt_model.py::_postprocess but
            inlines the MTP loss loop (using the *unmodified* CLM for real
            per-token CE) and returns logits in [b, s, h] form unconditionally
            on the post_process branch -- never returning a CE loss tensor
            to AReaL's RL pipeline.

            v18: emits per-layer step-level metrics via stats_tracker so the
            trainer log surfaces actionable MTP signal:
              * mtp/layer{i}/loss_avg     -- token-mean CE per MTP layer
              * mtp/layer{i}/token_acc    -- top-1 accuracy on valid tokens
              * mtp/layer{i}/num_valid    -- # tokens contributing to loss
              * mtp/scaling_factor        -- echo of mtp_loss_scaling_factor
            All emissions are wrapped in try/except so stat failures never
            break training.
            """
            in_inference_mode = (
                inference_context is not None and not self_model.training
            )
            if in_inference_mode:
                assert runtime_gather_output, (
                    "Inference must always gather TP logits"
                )

            if is_spec_decode is None:
                try:
                    is_spec_decode = (
                        in_inference_mode
                        and inference_context.is_dynamic_batching()
                        and inference_context.num_speculative_tokens > 0
                    )
                except Exception:  # pragma: no cover
                    is_spec_decode = False

            if not self_model.training or in_inference_mode or is_spec_decode:
                return _orig_unbound(
                    self_model,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    labels=labels,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    mtp_in_postprocess=mtp_in_postprocess,
                    loss_mask=loss_mask,
                    decoder_input=decoder_input,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    runtime_gather_output=runtime_gather_output,
                    extra_block_kwargs=extra_block_kwargs,
                    inference_context=inference_context,
                    is_spec_decode=is_spec_decode,
                )

            output_weight = None
            if self_model.share_embeddings_and_output_weights:
                output_weight = (
                    self_model.shared_embedding_or_output_weight()
                )

            # ---- Run MTP block ----
            if mtp_in_postprocess:
                # ----------------------------------------------------------
                # v14: bypass megatron-core 0.17.0's MTP recompute checkpoint
                # which is incompatible with packed_seq_params.
                #
                # Root cause:
                #   When `config.recompute_granularity == 'full'` AND
                #   `self.training` is True, `MultiTokenPredictionLayer.forward`
                #   routes through `self._checkpointed_forward` which calls
                #   `tensor_parallel.checkpoint(forward_func, ...)`.
                #   `tensor_parallel.checkpoint` -> `CheckpointFunction.apply`
                #   internally `ctx.save_for_backward(*args)` which PyTorch
                #   restricts to `torch.Tensor` only. `PackedSeqParams` is a
                #   dataclass (qkv_format str + ints + tensors), so
                #   `save_for_backward` raises TypeError.
                #
                # Fix:
                #   Temporarily set `recompute_granularity = None` on every
                #   config object reachable from `self_model.mtp` for the
                #   duration of the MTP block forward. Restored in `finally`.
                #
                # Slime alignment:
                #   slime under THUDM Megatron does NOT enable MTP recompute
                #   by default. This patch replicates that behaviour.
                # ----------------------------------------------------------
                _saved_recompute = []
                try:
                    _seen_cfg_ids = set()

                    def _stash_cfg(cfg):
                        if cfg is None:
                            return
                        if id(cfg) in _seen_cfg_ids:
                            return
                        _seen_cfg_ids.add(id(cfg))
                        if not hasattr(cfg, "recompute_granularity"):
                            return
                        _saved_recompute.append(
                            (cfg, cfg.recompute_granularity)
                        )
                        try:
                            cfg.recompute_granularity = None
                        except Exception:
                            # If the attribute is read-only on the dataclass,
                            # leave it as-is; layer-level configs (which are
                            # actually consulted) are mutable.
                            pass

                    _mtp_module = self_model.mtp
                    _stash_cfg(getattr(_mtp_module, "config", None))
                    _layers = getattr(_mtp_module, "layers", None)
                    if _layers is not None:
                        for _ml in _layers:
                            _stash_cfg(getattr(_ml, "config", None))
                            for _child_attr in (
                                "transformer_layer",
                                "mtp_model_layer",
                            ):
                                _child = getattr(_ml, _child_attr, None)
                                if _child is not None:
                                    _stash_cfg(
                                        getattr(_child, "config", None)
                                    )

                    hidden_states = self_model.mtp(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        inference_params=None,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        embedding=self_model.embedding,
                        **(extra_block_kwargs or {}),
                    )
                finally:
                    # Always restore, even on exception, so subsequent
                    # iterations / non-MTP forwards see the original
                    # recompute settings.
                    for _cfg_r, _prev in _saved_recompute:
                        try:
                            _cfg_r.recompute_granularity = _prev
                        except Exception:
                            pass

            if not self_model.post_process:
                return hidden_states

            # ---- Inline MTP loss loop ----
            if _mtp_num_layers > 0 and labels is not None:
                # Validate divisibility BEFORE chunk to surface a clean error.
                _S = hidden_states.shape[0]
                if _S % (1 + _mtp_num_layers) != 0:
                    raise RuntimeError(
                        f"[AReaL][MTP] MTP block output dim 0 ({_S}) is not "
                        f"divisible by 1+mtp_num_layers={1 + _mtp_num_layers}. "
                        "self_model.mtp did not produce the expected (1+N)*S "
                        "layout. Check that mtp_in_postprocess=True actually "
                        "triggered the MTP block."
                    )
                hidden_states_list = torch.chunk(
                    hidden_states, 1 + _mtp_num_layers, dim=0
                )
                hidden_states = hidden_states_list[0]

                mtp_labels = labels.clone()
                if loss_mask is None:
                    _mtp_loss_mask = torch.ones_like(mtp_labels)
                else:
                    _mtp_loss_mask = loss_mask
                original_num_tokens = _mtp_loss_mask.sum()
                _cp_group = getattr(
                    getattr(self_model, "pg_collection", None),
                    "cp", None,
                )

                # v18: emit scaling factor once per forward (cheap, idempotent).
                try:
                    stats_tracker.scalar(
                        **{"mtp/scaling_factor": float(_mtp_loss_scaling_factor)}
                    )
                except Exception:
                    pass

                for mtp_layer_number in range(_mtp_num_layers):
                    mtp_logits, _ = self_model.output_layer(
                        hidden_states_list[mtp_layer_number + 1],
                        weight=output_weight,
                        runtime_gather_output=runtime_gather_output,
                    )
                    if getattr(_cfg, "use_mup", False):
                        try:
                            mtp_logits = self_model._scale_logits(mtp_logits)
                        except Exception:
                            pass

                    mtp_labels, _ = roll_tensor(
                        mtp_labels,
                        shifts=-1,
                        dims=-1,
                        cp_group=_cp_group,
                        packed_seq_params=packed_seq_params,
                    )
                    _mtp_loss_mask, num_tokens = roll_tensor(
                        _mtp_loss_mask,
                        shifts=-1,
                        dims=-1,
                        cp_group=_cp_group,
                        packed_seq_params=packed_seq_params,
                    )
                    mtp_loss = self_model.compute_language_model_loss(
                        mtp_labels, mtp_logits
                    )
                    mtp_loss = _mtp_loss_mask * mtp_loss

                    # ----------------------------------------------------------
                    # Per-MTP-layer loss reporting (slime alignment).
                    # ----------------------------------------------------------
                    try:
                        from megatron.core.transformer.multi_token_prediction import (
                            MTPLossLoggingHelper,
                        )
                        _mtp_loss_for_log = (
                            torch.sum(mtp_loss) / num_tokens
                            if num_tokens > 0
                            else mtp_loss.new_tensor(0.0)
                        )
                        MTPLossLoggingHelper.save_loss_to_tracker(
                            _mtp_loss_for_log,
                            mtp_layer_number,
                            _mtp_num_layers,
                            avg_group=mpu.get_data_parallel_group(
                                with_context_parallel=True
                            ),
                        )
                    except Exception:
                        pass

                    # ----------------------------------------------------------
                    # v18: AReaL-side per-MTP-layer step-level metrics.
                    #
                    # Why these matter for diagnosing slow accept_rate growth:
                    #   * mtp/layer{i}/loss_avg trending DOWN every step is the
                    #     necessary condition for the draft head to learn. Flat
                    #     or noisy curves => effective lr / scaling too small.
                    #   * mtp/layer{i}/token_acc is the *direct* offline proxy
                    #     for SGLang's online accept_rate. If offline acc is
                    #     ~0.6+ but online accept_rate stalls at 0.3-0.4, the
                    #     bottleneck is online (e.g. weight sync staleness,
                    #     EAGLE topk too narrow).
                    #   * mtp/layer{i}/num_valid surfaces token attrition due
                    #     to the per-layer roll (each layer loses 1 valid
                    #     token at the right edge); a sudden drop reveals
                    #     loss_mask shape mismatches.
                    # All emissions are wrapped: failures NEVER break training.
                    # ----------------------------------------------------------
                    try:
                        with torch.no_grad():
                            _num_valid = float(num_tokens.detach().item()) \
                                if hasattr(num_tokens, "detach") \
                                else float(num_tokens)
                            _loss_sum = float(mtp_loss.detach().sum().item())
                            _loss_avg = (
                                _loss_sum / _num_valid
                                if _num_valid > 0
                                else 0.0
                            )
                            # ----------------------------------------------
                            # v19 BUGFIX: token-level top-1 accuracy.
                            #
                            # Megatron logits convention is sequence-first
                            # ``[s, b, vocab]`` (the same layout returned by
                            # ``output_layer`` above). Labels and the loss
                            # mask, however, are batch-first ``[b, s]``
                            # (consistent with ``compute_language_model_loss``
                            # which is what produced ``mtp_loss`` here, also
                            # ``[b, s]``).
                            #
                            # In v18 we did:
                            #   _pred  = mtp_logits.argmax(dim=-1)   # [s,b]
                            #   match  = (_pred == mtp_labels)       # ←broadcast bug
                            #   total  = (match * loss_mask).sum()
                            # When ``s == b`` PyTorch silently broadcasts
                            # ``[s,b]`` against ``[b,s]`` to ``[?, ?]`` and
                            # the running mask broadcast inflates the sum
                            # by a factor of ~b/s. The exported metric in
                            # spec_v1.log.15 hit values like 48–407 instead
                            # of the expected ``[0, 1]`` range, confirming
                            # this is exactly the bug.
                            #
                            # Fix: bring logits into ``[b, s, vocab]`` BEFORE
                            # argmax so all three tensors agree on ``[b, s]``.
                            #
                            # TP > 1 caveat: when the LM head uses TP-sharded
                            # output (i.e. ``runtime_gather_output`` is False)
                            # the local logits cover only ``vocab/tp`` and
                            # local argmax is meaningless. We detect that by
                            # comparing the logits' last dim against the
                            # configured vocab; if smaller, we skip
                            # ``token_acc`` and only emit ``loss_avg`` /
                            # ``num_valid`` to avoid reporting garbage.
                            # Slime's MTP reporter uses ``loss_for_log`` only
                            # for the same reason — they do NOT export an
                            # offline token-acc — so silently dropping it on
                            # TP > 1 stays in alignment.
                            # ----------------------------------------------
                            _pred_logits = mtp_logits.detach()
                            _local_vocab = int(_pred_logits.shape[-1])
                            try:
                                _full_vocab = int(getattr(_cfg, "vocab_size",
                                                          _local_vocab))
                            except Exception:
                                _full_vocab = _local_vocab
                            _logits_are_full = (
                                _local_vocab == _full_vocab
                            )
                            _stats_payload = {
                                f"mtp/layer{mtp_layer_number}/loss_avg":
                                    _loss_avg,
                                f"mtp/layer{mtp_layer_number}/num_valid":
                                    _num_valid,
                            }
                            if _logits_are_full and _num_valid > 0:
                                # Transpose to [b, s, vocab] then argmax.
                                _pred = (
                                    _pred_logits.transpose(0, 1)
                                    .argmax(dim=-1)
                                )
                                _labels = mtp_labels.detach()
                                _mask = _mtp_loss_mask.detach()
                                if _pred.shape != _labels.shape:
                                    raise RuntimeError(
                                        "[AReaL][MTP] token_acc shape "
                                        f"mismatch: pred={tuple(_pred.shape)} "
                                        f"labels={tuple(_labels.shape)}"
                                    )
                                _match = (_pred == _labels).to(_mask.dtype)
                                _correct = float(
                                    (_match * _mask).sum().item()
                                )
                                _token_acc = _correct / _num_valid
                                _stats_payload[
                                    f"mtp/layer{mtp_layer_number}/token_acc"
                                ] = _token_acc
                            stats_tracker.scalar(**_stats_payload)
                    except Exception as _stat_exc:  # pragma: no cover
                        # Defensive: do NOT let stat aggregation kill training.
                        logger.debug(
                            "[AReaL][MTP] step-level stats emission failed "
                            "for layer %d: %r",
                            mtp_layer_number,
                            _stat_exc,
                        )

                    mtp_loss_scale = (
                        _mtp_loss_scaling_factor / max(1, _mtp_num_layers)
                    )
                    if _calculate_per_token_loss:
                        num_tokens_safe = torch.clamp(num_tokens, min=1)
                        mtp_loss_normalized = (
                            mtp_loss_scale
                            * mtp_loss
                            * (original_num_tokens / num_tokens_safe)
                        )
                        hidden_states = MTPLossAutoScaler.apply(
                            hidden_states, mtp_loss_normalized
                        )
                    else:
                        safe_num_tokens = num_tokens.clamp(min=1)
                        hidden_states = MTPLossAutoScaler.apply(
                            hidden_states,
                            mtp_loss_scale * mtp_loss / safe_num_tokens,
                        )

            # ---- Output layer + return logits ----
            logits, _ = self_model.output_layer(
                hidden_states,
                weight=output_weight,
                runtime_gather_output=runtime_gather_output,
            )

            try:
                logits = self_model._scale_logits(logits)
            except Exception:
                pass

            return logits.transpose(0, 1).contiguous()

        _patched_postprocess._areal_mtpaware_v12_replacement = True

        try:
            inner._postprocess = _types_mod.MethodType(
                _patched_postprocess, inner
            )
            self._patched_postprocess = True
            self._activated = True
            logger.debug(
                "[AReaL][MTP] training: replaced _postprocess "
                "(mtp_num_layers=%d, scaling=%g, calculate_per_token_loss=%s); "
                "extra_model_kwargs={labels: %s, loss_mask: %s}.",
                _mtp_num_layers,
                _mtp_loss_scaling_factor,
                _calculate_per_token_loss,
                tuple(labels.shape),
                "yes" if loss_mask is not None else "none",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[AReaL][MTP] training: failed to install _postprocess "
                "replacement (%s); MTP loss will NOT be injected.",
                exc,
            )
            try:
                if self._patched_postprocess and self._orig_postprocess is not None:
                    inner._postprocess = self._orig_postprocess
            except Exception:  # pragma: no cover - defensive
                pass
            self._patched_postprocess = False
            self.extra_model_kwargs = {}

    def __exit__(self, exc_type, exc, tb) -> None:
        inner = self._inner
        if inner is None or not self._activated:
            return
        try:
            if self._forward_only:
                # Restore inference-mode disable.
                if self._had_mtp_attr:
                    inner.mtp = self._saved_mtp
                else:
                    try:
                        delattr(inner, "mtp")
                    except AttributeError:
                        pass
                if self._had_mtp_process_attr:
                    inner.mtp_process = self._saved_mtp_process
                else:
                    try:
                        delattr(inner, "mtp_process")
                    except AttributeError:
                        pass
                cfg = getattr(inner, "config", None)
                if cfg is not None and self._had_mtp_num_layers_attr:
                    cfg.mtp_num_layers = self._saved_mtp_num_layers
                try:
                    delattr(inner, "_areal_mtpaware_disabled")
                except AttributeError:
                    pass
            else:
                # Restore training-mode patch.
                # v12: only _postprocess is replaced (no CLM patch).
                # v11-compat: if _patched_compute_lm_loss happens to be True
                # because of a stale code path, restore CLM too -- harmless.
                if self._patched_compute_lm_loss:
                    try:
                        inner.compute_language_model_loss = (
                            self._orig_compute_lm_loss
                        )
                    except Exception:  # pragma: no cover - defensive
                        pass
                # v12: restore the bound _postprocess we replaced.
                if self._patched_postprocess:
                    try:
                        inner._postprocess = self._orig_postprocess
                    except Exception:  # pragma: no cover - defensive
                        pass
        finally:
            logger.debug("[AReaL][MTP][v12] forward orchestrator restored.")


def all_reduce_mtp_loss(loss: torch.Tensor) -> torch.Tensor:
    """All-reduce the MTP loss across the data-parallel group, then return the
    averaged scalar (used purely for logging — does not affect gradients).
    """
    if not torch.is_tensor(loss):
        return loss
    out = loss.detach().clone()
    try:
        dp_group = mpu.get_data_parallel_group()
        dist.all_reduce(out, group=dp_group)
        out /= max(1, dist.get_world_size(group=dp_group))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[AReaL][MTP] all_reduce_mtp_loss skipped: {exc}")
    return out


# ---------------------------------------------------------------------------
# Hidden-state capture (forward hook)
# ---------------------------------------------------------------------------


class MTPHiddenStateCapturer:
    """Register a forward hook on the inner GPTModel's final_layernorm that
    captures pre-output-layer hidden states. The MTP block needs these
    hidden states (NOT logits) as one of its inputs.

    By capturing externally we avoid having to fork or patch upstream Megatron.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handle = None
        self.last_hidden: Optional[torch.Tensor] = None
        self._inner = _unwrap_to_gpt_model(model)

    def _hook(self, module, inputs, output):
        # Megatron's final_layernorm returns the normed hidden states.
        if isinstance(output, tuple):
            output = output[0]
        self.last_hidden = output

    def __enter__(self):
        if not model_has_mtp(self.model):
            return self
        decoder = getattr(self._inner, "decoder", None)
        if decoder is None:
            print("[AReaL][MTP] capturer: no decoder, skip hook")
            return self
        final_ln = getattr(decoder, "final_layernorm", None)
        if final_ln is None:
            print("[AReaL][MTP] capturer: no final_layernorm, skip hook")
            return self
        self.handle = final_ln.register_forward_hook(self._hook)
        print("[AReaL][MTP] capturer: forward hook installed on final_layernorm")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            print("[AReaL][MTP] capturer: forward hook removed")
        self.handle = None

    def get(self) -> Optional[torch.Tensor]:
        return self.last_hidden


# ---------------------------------------------------------------------------
# mbridge MimoBridge compatibility patch (HF -> Megatron weight load)
# ---------------------------------------------------------------------------
# Newer Megatron-Core revisions renamed the inner MTP submodule from
# ``transformer_layer`` to ``mtp_model_layer``. The vendored
# ``mbridge.MimoBridge._convert_mtp_param`` only accepts the legacy name and
# raises ``NotImplementedError`` on anything else, e.g.
#   ``mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight``
# crashes the entire HF -> Megatron weight load.
#
# This installer is borrowed from
# https://github.com/areal-project/AReaL/pull/1176 (file
# ``areal/models/mcore/mimo_mtp_hf_mapping.py``). PR#1176 augments
# ``local_to_hf_map`` AFTER the bridge has populated it, which assumes the
# bridge returns an empty list for unknown MTP keys. Because our installed
# mbridge instead RAISES, we must intercept earlier — by monkey-patching
# ``MimoBridge._weight_name_mapping_mcore_to_hf`` to short-circuit MTP names
# through our own resolver before the unsupported branch is reached.
#
# The patch is idempotent (guarded by ``_areal_mtp_patched``) and silently
# no-ops if mbridge / MimoBridge / our mapping helper isn't importable.

def install_mbridge_mtp_compat_patch():
    """Monkey-patch ``MimoBridge._weight_name_mapping_mcore_to_hf`` to accept
    both ``transformer_layer`` and ``mtp_model_layer`` MTP submodule names.

    Safe to call multiple times. MUST be called before any code that triggers
    HF -> Megatron weight loading (i.e. before
    ``load_weights_from_hf_with_mbridge_fast``)."""
    try:
        from mbridge.models.mimo import MimoBridge  # type: ignore
    except ImportError:
        logger.info(
            "[AReaL][MTP][compat] mbridge.models.mimo not importable; "
            "skipping MimoBridge MTP compat patch."
        )
        return
    try:
        from areal.models.mcore.mimo_mtp_hf_mapping import (
            mtp_mcore_name_to_hf_names,
        )
    except ImportError as e:
        logger.warning(
            "[AReaL][MTP][compat] mimo_mtp_hf_mapping unavailable (%s); "
            "skipping MimoBridge MTP compat patch.",
            e,
        )
        return

    if getattr(MimoBridge, "_areal_mtp_patched", False):
        return

    orig = MimoBridge._weight_name_mapping_mcore_to_hf

    def patched(self, mcore_weights_name):
        # Fast path: only MTP names need our intervention. For everything else
        # delegate to the original method (which already handles the base
        # decoder, output_layer, etc.).
        if "mtp" in mcore_weights_name:
            hf_names = mtp_mcore_name_to_hf_names(mcore_weights_name)
            if hf_names:
                return hf_names
            # _extra_state and similar non-tensor entries: emit empty list so
            # the caller skips them instead of raising NotImplementedError.
            if mcore_weights_name.endswith("_extra_state"):
                return []
        try:
            return orig(self, mcore_weights_name)
        except NotImplementedError:
            # The legacy bridge raises for any MTP name it doesn't recognise.
            # Try once more via our resolver (covers the case where 'mtp' is
            # not literally in the name but the regex still matches).
            hf_names = mtp_mcore_name_to_hf_names(mcore_weights_name)
            if hf_names:
                return hf_names
            raise

    MimoBridge._weight_name_mapping_mcore_to_hf = patched
    MimoBridge._areal_mtp_patched = True
    logger.info(
        "[AReaL][MTP][compat] Installed MimoBridge._weight_name_mapping_"
        "mcore_to_hf compat patch (accepts both transformer_layer and "
        "mtp_model_layer MTP submodule naming)."
    )


# ---------------------------------------------------------------------------
# SGLang MiMoMTP EAGLE3 compat patch
# ---------------------------------------------------------------------------
def install_mimomtp_eagle3_compat_patch():
    """Add a no-op-style ``set_eagle3_layers_to_capture`` to SGLang's ``MiMoMTP``.

    Why this is needed
    ------------------
    When the user launches SGLang with ``--speculative-algorithm EAGLE3`` and
    MTP-as-draft (i.e. the draft model is ``MiMoMTP`` from
    ``sglang.srt.models.mimo_mtp``), ``CudaGraphRunner.__init__`` runs the
    following block during scheduler startup::

        if model_runner.spec_algorithm.is_eagle3():
            self.model_runner.model.set_eagle3_layers_to_capture()

    Upstream EAGLE3-aware backbones (``Qwen2ForCausalLM``,
    ``LlamaForCausalLM``, ``DeepseekV2ForCausalLM``, ...) all implement this
    method.  ``MiMoMTP`` does not, so the call raises::

        AttributeError: 'MiMoMTP' object has no attribute 'set_eagle3_layers_to_capture'

    The crash happens *before* the scheduler can run a single step, killing
    the whole inference process and tearing down the AReaL trainer.

    What the stub does
    ------------------
    ``MiMoMTP`` is the *draft* head, not the target.  EAGLE3's "aux hidden
    state capture" is a property of the *target* model (it has to expose
    intermediate decoder layer outputs via ``layers_to_capture``).  An MTP
    head only consumes a single hidden state via
    ``forward_batch.spec_info.hidden_states`` and produces one output token
    per call -- there are no aux layers to capture from a single-block draft.

    Therefore the correct stub:
      * sets ``capture_aux_hidden_states = False`` on the MTP module so any
        downstream attribute probe (``getattr(model,
        'capture_aux_hidden_states', ...)``) returns a sane value;
      * leaves the MTP module's internal layer wiring untouched (there is no
        ``self.model.layers`` ModuleList to slice -- ``MiMoMultiTokenPredictor
        Layer`` has only one ``mtp_block``);
      * stores an empty ``layers_to_capture = []`` sentinel on
        ``self.model`` so any later code that does ``model.model.layers_to_
        capture`` gets a list-typed value instead of ``AttributeError``;
      * accepts an optional ``layer_ids`` argument to match the signature of
        the upstream method exactly (forward-compat for callers that pass
        explicit layer ids).

    This is intentionally *not* a no-op: SGLang's ``CudaGraphRunner``
    inspects ``capture_aux_hidden_states`` later when wiring up the
    ``capture_hidden_mode``; we must explicitly mark it ``False`` to keep the
    runner on the FULL hidden-state path that MTP requires.

    Idempotency
    -----------
    Safe to call multiple times.  Uses ``MiMoMTP._areal_eagle3_patched`` as a
    sentinel so repeated invocations are no-ops.

    Where to call
    -------------
    MUST be called *before* SGLang's ``Scheduler(...)`` is instantiated in
    each worker process.  In AReaL the call site is
    ``areal/experimental/inference_service/sglang/scheduler.py::
    areal_run_scheduler_process`` immediately before the ``Scheduler(...)``
    constructor.
    """
    try:
        from sglang.srt.models.mimo_mtp import MiMoMTP  # type: ignore
    except ImportError:
        logger.info(
            "[AReaL][MTP][compat] sglang.srt.models.mimo_mtp not importable; "
            "skipping MiMoMTP EAGLE3 compat patch."
        )
        return

    if getattr(MiMoMTP, "_areal_eagle3_patched", False):
        return

    def set_eagle3_layers_to_capture(self, layer_ids=None):
        # MTP-as-draft: no aux hidden state capture is meaningful here.
        # Marking capture_aux_hidden_states=False keeps SGLang's
        # CudaGraphRunner on the FULL hidden-state path that MTP needs.
        self.capture_aux_hidden_states = False
        # Provide a list-typed sentinel so downstream code that probes
        # ``model.model.layers_to_capture`` does not AttributeError.
        try:
            self.model.layers_to_capture = []
        except Exception:
            # ``self.model`` is always a MiMoMultiTokenPredictorLayer; the
            # ``except`` is purely defensive against future refactors.
            pass
        # Accept ``layer_ids`` solely to mirror the upstream signature; we
        # deliberately ignore it because MiMoMTP has only a single mtp_block
        # and no aux layers to slice.
        del layer_ids

    MiMoMTP.set_eagle3_layers_to_capture = set_eagle3_layers_to_capture
    MiMoMTP._areal_eagle3_patched = True
    logger.info(
        "[AReaL][MTP][compat] Installed MiMoMTP.set_eagle3_layers_to_capture "
        "stub (EAGLE3 + MTP-as-draft compatibility)."
    )


# ---------------------------------------------------------------------------
# SGLang MiMoMultiTokenPredictorLayer.forward spec_info compat patch
# ---------------------------------------------------------------------------
def install_mimomtp_spec_info_compat_patch():
    """Bridge ``EagleVerifyInput`` -> ``hidden_states`` for ``MiMoMTP`` draft.

    Why this is needed
    ------------------
    Upstream ``MiMoMultiTokenPredictorLayer.forward`` (sglang/srt/models/
    mimo_mtp.py) reads ``forward_batch.spec_info.hidden_states`` unconditionally::

        hidden_states = self.eh_proj(
            torch.cat(
                (
                    forward_batch.spec_info.hidden_states,
                    self.enorm(self.embed_tokens(input_ids)),
                ),
                dim=-1,
            )
        )

    SGLang has two distinct spec_info classes:
      * ``EagleDraftInput`` -- has the ``hidden_states: torch.Tensor`` field
        and is used during the *draft* stage.
      * ``EagleVerifyInput`` -- does NOT have ``hidden_states`` and is used
        during the *verify* stage (also during cuda graph capture in some
        spec_algorithm modes).

    When SGLang reaches the MTP draft head with an ``EagleVerifyInput`` (e.g.
    during cuda graph capture under EAGLE3 mode, or in any future code path
    that reuses verify inputs for draft eval), it raises::

        AttributeError: 'EagleVerifyInput' object has no attribute 'hidden_states'

    PR#1176 in inclusionAI/AReaL documents EAGLE-only (not EAGLE3) usage with
    the built-in MiMoMTP draft head.  The cli_args.py auto-downgrade
    (EAGLE3->EAGLE) is the primary fix; this patch is the *defensive*
    second-layer fix for any code path that still reaches MiMoMTP with a
    non-draft spec_info.

    What the patch does
    -------------------
    Wraps ``MiMoMultiTokenPredictorLayer.forward`` so that:

    1. If ``forward_batch.spec_info`` is ``None`` or already exposes
       ``hidden_states`` (the EAGLE happy path), the original forward runs
       unchanged.
    2. Otherwise, we look up a fallback hidden state in this priority order:
         a. ``forward_batch.hidden_states``        (target model's last layer)
         b. ``forward_batch.hidden_states_backup`` (cuda-graph backup buffer)
       and temporarily set it onto ``spec_info.hidden_states`` for the
       duration of the call, restoring the prior state in a ``finally`` block.
    3. If no fallback is available, raises a clear error message that points
       at the EAGLE3-vs-EAGLE configuration mismatch (the actionable fix is
       to use ``speculative_algorithm: "EAGLE"`` per PR#1176).

    Idempotency
    -----------
    Safe to call multiple times.  Uses
    ``MiMoMultiTokenPredictorLayer._areal_spec_info_patched`` as sentinel.

    Where to call
    -------------
    MUST be called *before* SGLang's ``Scheduler(...)`` is instantiated in
    each worker process, alongside ``install_mimomtp_eagle3_compat_patch``.
    """
    try:
        from sglang.srt.models.mimo_mtp import (  # type: ignore
            MiMoMultiTokenPredictorLayer,
        )
    except ImportError:
        logger.info(
            "[AReaL][MTP][compat] sglang.srt.models.mimo_mtp not importable; "
            "skipping MiMoMultiTokenPredictorLayer spec_info compat patch."
        )
        return

    if getattr(
        MiMoMultiTokenPredictorLayer, "_areal_spec_info_patched", False
    ):
        return

    _orig_forward = MiMoMultiTokenPredictorLayer.forward

    def _resolve_fallback_hidden_states(forward_batch):
        # Priority 1: target model's last hidden states (set by the verifier
        # before invoking the draft head in PR#1176's flow).
        hs = getattr(forward_batch, "hidden_states", None)
        if hs is not None:
            return hs, "forward_batch.hidden_states"
        # Priority 2: cuda-graph backup buffer.
        hs = getattr(forward_batch, "hidden_states_backup", None)
        if hs is not None:
            return hs, "forward_batch.hidden_states_backup"
        return None, None

    def patched_forward(
        self, input_ids, positions, forward_batch, input_embeds=None
    ):
        spec_info = getattr(forward_batch, "spec_info", None)
        # Happy path: EagleDraftInput with hidden_states already populated.
        if spec_info is not None and getattr(
            spec_info, "hidden_states", None
        ) is not None:
            return _orig_forward(
                self, input_ids, positions, forward_batch, input_embeds
            )
        # Defensive bridge: spec_info missing hidden_states (e.g.
        # EagleVerifyInput reached during cuda graph capture).
        bridged_hs, source = _resolve_fallback_hidden_states(forward_batch)
        if bridged_hs is None:
            raise AttributeError(
                "[AReaL][MTP][compat] MiMoMultiTokenPredictorLayer.forward "
                "could not find a hidden_states tensor on spec_info "
                f"({type(spec_info).__name__ if spec_info else 'None'}) "
                "nor on forward_batch.hidden_states / "
                "forward_batch.hidden_states_backup. This usually means the "
                "scheduler is using speculative_algorithm='EAGLE3' with the "
                "built-in MiMoMTP draft head, which is incompatible per "
                "PR#1176. Set speculative_algorithm='EAGLE' to fix."
            )
        if spec_info is None:
            # Cannot stash hidden_states without a spec_info container.
            raise AttributeError(
                "[AReaL][MTP][compat] forward_batch.spec_info is None; "
                "MiMoMultiTokenPredictorLayer requires a spec_info object."
            )
        _had_attr = hasattr(spec_info, "hidden_states")
        _prev = getattr(spec_info, "hidden_states", None) if _had_attr else None
        try:
            # Temporarily inject hidden_states so the original forward sees
            # the EAGLE-shaped contract.
            try:
                setattr(spec_info, "hidden_states", bridged_hs)
            except (AttributeError, TypeError) as exc:
                # Slotted dataclasses may reject setattr; surface a clear
                # error rather than crashing inside _orig_forward.
                raise AttributeError(
                    "[AReaL][MTP][compat] cannot set "
                    f"hidden_states on {type(spec_info).__name__}: {exc}. "
                    "Use speculative_algorithm='EAGLE' (PR#1176) instead of "
                    "'EAGLE3' with the built-in MiMoMTP draft head."
                ) from exc
            logger.debug(
                "[AReaL][MTP][compat] Bridged spec_info.hidden_states from %s "
                "(spec_info=%s).",
                source,
                type(spec_info).__name__,
            )
            return _orig_forward(
                self, input_ids, positions, forward_batch, input_embeds
            )
        finally:
            if _had_attr:
                try:
                    setattr(spec_info, "hidden_states", _prev)
                except Exception:  # pragma: no cover - defensive
                    pass
            else:
                try:
                    delattr(spec_info, "hidden_states")
                except AttributeError:
                    pass

    MiMoMultiTokenPredictorLayer.forward = patched_forward
    MiMoMultiTokenPredictorLayer._areal_spec_info_patched = True
    logger.info(
        "[AReaL][MTP][compat] Installed MiMoMultiTokenPredictorLayer.forward "
        "spec_info bridge (EagleVerifyInput -> forward_batch.hidden_states "
        "fallback for cuda graph capture under EAGLE3 mode)."
    )
