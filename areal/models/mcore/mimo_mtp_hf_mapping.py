"""MiMo MTP HF name-mapping helper (AReaL EAGLE+MTP, v4).

Why this module exists
----------------------
The upstream ``mbridge.MimoBridge`` only knows MTP parameter names of the form
``mtp.layers.{i}.transformer_layer.<...>`` and raises
``NotImplementedError`` for anything else. Newer Megatron-Core revisions
rename the inner submodule from ``transformer_layer`` to ``mtp_model_layer``,
so a freshly-trained MiMo MTP head produces names like
``mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight`` which
the bridge cannot translate. The result is a hard crash inside
``MimoBridge._convert_mtp_param`` during HF→Megatron weight load.

This module ports the pure-data mapping table from
`areal-project/AReaL#1176 <https://github.com/areal-project/AReaL/pull/1176>`_
(file ``areal/models/mcore/mimo_mtp_hf_mapping.py``) and extends it with a
``mtp_model_layer``↔``transformer_layer`` alias, so both Megatron submodule
naming conventions are accepted transparently.

Two integration paths are supported:

1. ``augment_local_to_hf_map_with_mtp`` — PR#1176's flow: invoked AFTER the
   bridge populates ``local_to_hf_map`` to authoritatively rewrite MTP rows.
   Currently unused by AReaL main but kept verbatim for forward compatibility.
2. ``mtp_mcore_name_to_hf_names`` — standalone resolver used by
   ``areal.engine.megatron_utils.mtp.install_mbridge_mtp_compat_patch`` to
   monkey-patch ``MimoBridge._weight_name_mapping_mcore_to_hf`` BEFORE it has
   a chance to raise on unknown MTP keys. This is the path that actually
   fixes the user's runtime crash; without it, the bridge's strict
   ``transformer_layer`` check rejects ``mtp_model_layer`` names outright.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

# Matches ``mtp.layers.{idx}.{rest}`` and the ``decoder.mtp_layers.{idx}.``
# variant that some megatron-core revisions emit.
_MTP_GLOBAL_RE = re.compile(
    r"^(?:decoder\.)?mtp(?:\.layers|_layers)\.(\d+)\.(.+)$"
)

# MCore MTP suffix  ->  HF suffix under ``model.mtp_layers.{idx}.``.
# Multi-valued entries are merged by the existing qkv / gate-up handling in
# ``hf_load._convert_hf_weights_to_mcore``.
_MTP_SUFFIX_MAP: Dict[str, object] = {
    # MTP-specific layer norms and projections
    "enorm.weight":           "token_layernorm.weight",
    "hnorm.weight":           "hidden_layernorm.weight",
    "eh_proj.weight":         "input_proj.weight",
    "final_layernorm.weight": "final_layernorm.weight",

    # transformer_layer.* (reused Qwen2 decoder block)
    "transformer_layer.input_layernorm.weight":
        "input_layernorm.weight",
    "transformer_layer.self_attention.linear_qkv.layer_norm_weight":
        "input_layernorm.weight",
    "transformer_layer.self_attention.linear_qkv.weight": [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ],
    "transformer_layer.self_attention.linear_qkv.bias": [
        "self_attn.q_proj.bias",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.bias",
    ],
    "transformer_layer.self_attention.linear_proj.weight":
        "self_attn.o_proj.weight",

    "transformer_layer.pre_mlp_layernorm.weight":
        "post_attention_layernorm.weight",
    "transformer_layer.mlp.linear_fc1.layer_norm_weight":
        "post_attention_layernorm.weight",
    "transformer_layer.mlp.linear_fc1.weight": [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ],
    "transformer_layer.mlp.linear_fc2.weight":
        "mlp.down_proj.weight",
}


def _normalize_mcore_submodule(rest: str) -> str:
    """Normalize the inner MTP submodule name.

    Newer Megatron-Core renamed ``transformer_layer`` to ``mtp_model_layer``;
    accept both transparently so the mapping table only needs one set of keys.
    """
    if rest.startswith("mtp_model_layer."):
        return "transformer_layer." + rest[len("mtp_model_layer."):]
    return rest


def mtp_mcore_name_to_hf_names(global_name: str) -> List[str]:
    """Return HF keys matching one MCore MTP-global name.

    Returns an empty list if ``global_name`` does not look like an MTP entry
    or has no explicit mapping rule (e.g. ``_extra_state`` tails, unknown
    subcomponents). Both ``transformer_layer.*`` and ``mtp_model_layer.*``
    inner naming variants are accepted.
    """
    m = _MTP_GLOBAL_RE.match(global_name)
    if m is None:
        return []
    idx, rest = m.group(1), m.group(2)
    if rest.endswith("_extra_state"):
        return []
    rest = _normalize_mcore_submodule(rest)
    rule = _MTP_SUFFIX_MAP.get(rest)
    if rule is None:
        return []
    prefix = f"model.mtp_layers.{idx}."
    if isinstance(rule, str):
        return [prefix + rule]
    return [prefix + s for s in rule]


def augment_local_to_hf_map_with_mtp(
    local_to_global_map: Dict[str, str],
    local_to_hf_map: Dict[str, List[str]],
    logger=None,
) -> int:
    """Inject MTP HF-name mappings into ``local_to_hf_map`` in-place.

    Ported verbatim from PR#1176 with ``mtp_model_layer`` alias support
    (handled inside ``mtp_mcore_name_to_hf_names``). Currently unused by
    AReaL main; kept here so future integrations can adopt the augment-after-
    bridge pattern without re-implementing the table.

    ``AREAL_MTP_P1_OVERWRITE=0`` reverts to "only fill empties" behaviour.

    Returns the number of local keys patched.
    """
    overwrite = os.environ.get("AREAL_MTP_P1_OVERWRITE", "1") == "1"
    patched = 0
    filled_empty = 0
    overwritten_nonempty = 0
    skipped_no_rule = 0
    preview_filled: List[str] = []
    preview_overwritten: List[str] = []
    for local_name, global_name in local_to_global_map.items():
        if "_extra_state" in local_name:
            continue
        m = _MTP_GLOBAL_RE.match(global_name)
        if m is None:
            continue
        hf_names = mtp_mcore_name_to_hf_names(global_name)
        if not hf_names:
            skipped_no_rule += 1
            continue
        cur = local_to_hf_map.get(local_name) or []
        if cur:
            if not overwrite:
                continue
            local_to_hf_map[local_name] = hf_names
            overwritten_nonempty += 1
            patched += 1
            if len(preview_overwritten) < 3:
                preview_overwritten.append(
                    f"{local_name}: {cur}->{hf_names}"
                )
        else:
            local_to_hf_map[local_name] = hf_names
            filled_empty += 1
            patched += 1
            if len(preview_filled) < 3:
                preview_filled.append(f"{local_name}->{hf_names}")
    if logger is not None:
        try:
            logger.info(
                "[AReaL][MTP][hf_load] augment_local_to_hf_map_with_mtp "
                "patched=%d (overwritten_nonempty=%d, filled_empty=%d, "
                "skipped_no_rule=%d) overwrite_mode=%s "
                "preview_overwritten=%s preview_filled=%s",
                patched, overwritten_nonempty, filled_empty,
                skipped_no_rule, overwrite,
                preview_overwritten, preview_filled,
            )
        except Exception:
            pass
    return patched
