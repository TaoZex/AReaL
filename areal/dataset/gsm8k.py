# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    data_source: str | None = None,
    data_source_split: list[str] | tuple[str, ...] | None = None,
):
    """Load GSM8K as an RL dataset.

    ``data_source`` (optional) stamps every sample with a routing identifier
    used by Multi-teacher On-Policy Distillation (MoPD) — see
    ``PPOConfig.teacher_key`` / ``PPOConfig.teachers``. Defaults to ``path``
    so the column is always populated and downstream RLVRWorkflow can
    propagate it onto each rollout trajectory unchanged.

    ``data_source_split`` (optional, mutually exclusive with ``data_source``)
    re-tags each row's routing key by sample index modulo the supplied
    list/tuple, e.g. ``["gsm8k:even", "gsm8k:odd"]`` deterministically
    splits GSM8K into two virtual sub-domains. This is the minimal way to
    exercise true multi-teacher MoPD routing without depending on a
    second external dataset; the underlying problem distribution and
    answer schema are unchanged so a single ``gsm8k_reward_fn`` still
    applies to every row.
    """
    if data_source is not None and data_source_split is not None:
        raise ValueError(
            "get_gsm8k_rl_dataset: 'data_source' and 'data_source_split' "
            "are mutually exclusive — pass at most one."
        )
    dataset = load_dataset(path=path, name="main", split=split)
    routing_key = data_source if data_source is not None else path
    split_keys: tuple[str, ...] | None = (
        tuple(data_source_split) if data_source_split is not None else None
    )
    if split_keys is not None and len(split_keys) < 2:
        raise ValueError(
            "get_gsm8k_rl_dataset: 'data_source_split' must list at least 2 "
            f"routing keys, got {split_keys!r}"
        )

    def process(sample, idx):
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        # ``data_source`` is the canonical routing field for MoPD.
        # When MoPD is disabled, downstream code simply ignores it.
        if split_keys is not None:
            row_key = split_keys[idx % len(split_keys)]
        else:
            row_key = routing_key
        return {"messages": messages, "data_source": row_key}

    dataset = dataset.map(process, with_indices=True).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
