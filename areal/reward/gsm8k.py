# SPDX-License-Identifier: Apache-2.0

import os
import sys
import threading

from areal.utils import logging

from . import get_math_verify_worker

logger = logging.getLogger("GSM8KReward")

# --- P2 diagnostic instrumentation -----------------------------------------
# In mopd.log.3, the ``torl_math`` per-source mean reward sat at 0.06–0.19
# while ``openai/gsm8k`` was 0.53–0.69. Both routes use this same function,
# so the gap is either (a) the model genuinely failing on harder ToRL math,
# or (b) a format mismatch making ``math_verify`` reject correct answers
# (e.g. ToRL ground-truth is ``\boxed{42}`` but the student emits a bare
# ``42``). Sampled logs below let us tell which.
#
# Per-source cap so the log volume stays bounded across a long run.
# ``_REWARD_LOG_CAP_PER_SOURCE`` samples per ``(source, verdict)`` cell get
# logged; after that, only counters increment. The cell-key is
# ``(data_source, success_bucket)`` where ``success_bucket`` is "ok" /
# "zero" / "exc" so we get diversity rather than just N copies of the first
# success.
_REWARD_LOG_LOCK = threading.Lock()
_REWARD_LOG_COUNTS: dict[tuple[str, str], int] = {}
_REWARD_LOG_CAP_PER_SOURCE = 5
_REWARD_HEAD_CHARS = 240


def _reward_diag(
    *,
    data_source: str,
    completion: str,
    answer: str,
    score: float,
    bucket: str,
    exc: BaseException | None,
) -> None:
    """Emit a sampled diagnostic line to stderr. Capped per
    ``(data_source, bucket)`` so a single training run produces at most
    a few dozen lines total."""
    key = (data_source or "<none>", bucket)
    with _REWARD_LOG_LOCK:
        n = _REWARD_LOG_COUNTS.get(key, 0)
        if n >= _REWARD_LOG_CAP_PER_SOURCE:
            return
        _REWARD_LOG_COUNTS[key] = n + 1
    try:
        head_c = completion[-_REWARD_HEAD_CHARS:].replace("\n", "\\n")
        head_a = answer[:_REWARD_HEAD_CHARS].replace("\n", "\\n")
        msg = (
            f"[gsm8k_reward][pid={os.getpid()}] ds={data_source!r} "
            f"bucket={bucket} score={score} "
            f"answer_head={head_a!r} completion_tail={head_c!r}"
        )
        if exc is not None:
            msg += f" exc={exc!r}"
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass


def gsm8k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    # ``data_source`` is forwarded through ``**kwargs`` by the workflow when
    # available (RLVRWorkflow plumbs the row dict in). Fall back to a literal
    # marker so the bucket key still groups missing values together.
    data_source = str(kwargs.get("data_source", "<unknown>"))
    completions_str = str(completions)
    answer_str = str(answer)
    try:
        worker = get_math_verify_worker()
        score = worker.verify(completions_str, answer_str)
    except Exception as e:
        logger.warning("Exception in gsm8k_reward_fn", exc_info=True)
        _reward_diag(
            data_source=data_source,
            completion=completions_str,
            answer=answer_str,
            score=0.0,
            bucket="exc",
            exc=e,
        )
        return 0.0
    # Bucket the result so we sample successes AND failures separately —
    # otherwise a source with a high success rate would only ever surface
    # ``ok`` examples and we'd never see the failure-case strings.
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    bucket = "ok" if score_f > 0 else "zero"
    _reward_diag(
        data_source=data_source,
        completion=completions_str,
        answer=answer_str,
        score=score,
        bucket=bucket,
        exc=None,
    )
    return score
