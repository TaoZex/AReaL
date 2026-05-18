# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
    ModelResponse,
    RolloutWorkflow,
)
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import apply_chat_template
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("RLVRWorkflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    return apply_chat_template(
        tokenizer,
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    return data["messages"]


class RLVRWorkflow(RolloutWorkflow):
    """Single-turn reward learning workflow supporting optional thinking tokens."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        get_input_ids_fn: Callable[[Any, PreTrainedTokenizerFast, bool], list[int]]
        | str = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any]
        | str = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        # Support string paths for get_input_ids_fn
        if isinstance(get_input_ids_fn, str):
            get_input_ids_fn = import_from_string(get_input_ids_fn)
        self.get_input_ids_fn = get_input_ids_fn
        # Support string paths for data_extract_prompt_fn
        if isinstance(data_extract_prompt_fn, str):
            data_extract_prompt_fn = import_from_string(data_extract_prompt_fn)
        self.data_extract_prompt_fn = data_extract_prompt_fn

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Decode completion and compute reward.

        Traces reward phase execution for SessionTracer. Decodes output tokens
        to string, calls async reward function, and logs metric to stats tracker.

        Returns
        -------
        float
            Reward value.
        """
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )

        return reward

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        """Generate one sample and compute its reward.

        Registers a new session for this sample, calls engine.agenerate,
        computes reward, and logs metrics. SessionTracer automatically
        tracks generate and reward phases via @trace_session decorators.

        Returns
        -------
        tuple[ModelResponse, float]
            Model response and reward value.
        """
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(resp, prompt_str, task_data)

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        return resp, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Generate single response and compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # Build result tensor dict with batch dim 1
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        out = {k: v.unsqueeze(0) for k, v in res.items()}
        # Propagate the optional ``data_source`` routing field so that
        # Multi-teacher On-Policy Distillation (MoPD) — which reads
        # ``traj[teacher_key]`` (default ``"data_source"``) — can route
        # this trajectory to the correct teacher engine, AND so that
        # downstream actor stats can split metrics (task_reward, correct
        # counts, sample counts) per data source.
        #
        # IMPORTANT: store as a *single-element list* rather than a plain
        # string. ``concat_padded_tensors`` (areal/utils/data.py:286-296)
        # has three branches per key:
        #   * tensor → ``_pad_cat_dim0``,
        #   * list   → ``[item for v in values for item in v]`` (correct
        #              per-trajectory concat),
        #   * else   → ``values[0]`` (silent collapse to traj[0]'s value!).
        # A bare string would hit the else-branch and the entire
        # mini-batch would inherit traj[0]'s data_source — which is
        # exactly what we need to avoid for per-source stats. Wrapping
        # in a list keeps every traj's routing key alive through concat.
        # The trainer's ``group_trajectories_by_teacher`` reads each
        # traj *before* concat (operating on the per-traj dicts), so
        # the list wrapping is transparent to MoPD routing — see
        # ``resolve_teacher_key`` in mopd_utils.py which simply does
        # ``sample[teacher_key]`` and treats the value as opaque.
        # ``group_trajectories_by_teacher`` happens to compare via ==
        # so a single-element list compares unequal to the bare string;
        # we therefore unwrap it back to a string for the trainer's
        # per-traj view (the list form is only needed *after* concat).
        if isinstance(data, dict) and "data_source" in data:
            out["data_source"] = [data["data_source"]]
        return out
