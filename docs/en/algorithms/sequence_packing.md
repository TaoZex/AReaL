# Sequence Packing Algorithms

AReaL supports configurable sequence packing algorithms for micro-batch
allocation during training. Sequence packing determines how variable-length
sequences are grouped into micro-batches, directly impacting load balance
across data-parallel (DP) ranks and overall training throughput.

## Supported Algorithms

| Algorithm | Key | Description | Complexity | Balance Quality |
|---|---|---|---|---|
| **First Fit Decreasing (FFD)** | `ffd` | Greedy bin-packing heuristic. Sorts sequences by length (descending) and assigns each to the first bin with remaining capacity. | O(n log n) | Good |
| **Karmarkar-Karp (KK)** | `kk` | Largest Differencing Method. Iteratively merges the two most imbalanced partial partitions using a max-heap, producing near-optimal balance. | O(n log n · k) | Excellent |

## Configuration

The packing algorithm is controlled by the `packing_algorithm` field in
`MicroBatchSpec`, which can be set directly in YAML configuration files.

### YAML Configuration

```yaml
# In your experiment config (e.g., examples/countdown/train_config.yaml)

actor:
  mb_spec:
    max_tokens_per_mb: 8192
    n_mbs: 4
    n_mbs_divisor: 1
    packing_algorithm: kk    # Options: "ffd" (default), "kk"
```

### Supported Fields in `mb_spec`

| Field | Type | Default | Description |
|---|---|---|---|
| `n_mbs` | `int` | `1` | Number of micro-batches (minimum if `max_tokens_per_mb` is set) |
| `granularity` | `int` | `1` | Grouping granularity for adjacent sequences |
| `max_tokens_per_mb` | `int \| None` | `None` | Maximum tokens per micro-batch |
| `n_mbs_divisor` | `int` | `1` | Final micro-batch count is rounded up to a multiple of this value |
| `packing_algorithm` | `str` | `"ffd"` | Sequence packing algorithm: `"ffd"` or `"kk"` |

### Python API

You can also set the algorithm programmatically:

```python
from areal.api.cli_args import MicroBatchSpec

# Using KK algorithm
mb_spec = MicroBatchSpec(
    max_tokens_per_mb=8192,
    n_mbs=4,
    packing_algorithm="kk",
)

# Or update an existing spec
mb_spec_kk = MicroBatchSpec.new(existing_spec, packing_algorithm="kk")
```

## When to Use KK

**Recommended scenarios for KK:**

- **Large-scale RL training** with highly variable sequence lengths (e.g., RLHF,
  PPO with open-ended generation). KK significantly reduces the spread between
  the most-loaded and least-loaded DP rank.
- **Bimodal sequence distributions** where a mix of very short and very long
  sequences makes greedy packing suboptimal.
- **High DP parallelism** (≥4 ranks), where even small load imbalances cause
  significant idle time due to synchronization barriers.

**When FFD is sufficient:**

- Uniform or near-uniform sequence lengths.
- Small-scale experiments where packing overhead matters more than balance.
- Latency-sensitive inference pipelines (FFD is slightly faster).

## Algorithm Details

### FFD (First Fit Decreasing)

1. Sort sequences by length in descending order.
2. For each sequence, assign it to the first bin whose current total plus the
   new sequence does not exceed `max_tokens_per_mb`.
3. If no bin has capacity, create a new bin.

FFD is a classic bin-packing heuristic with a worst-case approximation ratio
of 11/9 · OPT + 6/9 for the number of bins.

### KK (Karmarkar-Karp)

The Karmarkar-Karp algorithm (also called the Largest Differencing Method)
is a partition-balancing algorithm:

1. Create a singleton partition state for each sequence.
2. Push all states onto a max-heap ordered by **spread** (max partition sum −
   min partition sum).
3. Repeatedly pop the two states with the largest spread, merge them by
   pairing the heaviest partition of one with the lightest of the other, and
   push the merged state back.
4. The final state contains a near-optimal k-way partition.

For post-processing, any partition exceeding `max_tokens_per_mb` is split
greedily to respect capacity constraints.

**Theoretical guarantee:** For 2-way partitioning, KK achieves a residual
difference of O(n^{−Θ(log n)}), exponentially better than greedy approaches.

## Benchmark Results

On bimodal sequence distributions (30% long sequences in [800, 2048], 70% short
in [50, 200]), with 4 DP ranks and 200 sequences:

| Metric | FFD | KK | Improvement |
|---|---|---|---|
| Spread (max − min load) | ~757 tokens | ~8 tokens | **99% reduction** |
| Win rate (100 random trials) | 3% | **97%** | — |

> **Note:** Actual improvements depend on the specific sequence length
> distribution. KK's advantage is most pronounced with high-variance
> distributions.

## File Reference

| File | Change Description |
|---|---|
| `areal/utils/seqpack.py` | Added KK algorithm (`kk_allocate`, `_KKSet`, `_KKState`, `_kk_partition`), constants (`PACKING_ALGORITHM_FFD`, `PACKING_ALGORITHM_KK`, `PACKING_ALGORITHMS`), and `get_allocate_fn()` registry |
| `areal/api/cli_args.py` | Added `packing_algorithm` field to `MicroBatchSpec` with validation |
| `areal/utils/data.py` | Modified `allocate_balanced_mbs()` to dispatch via `get_allocate_fn(mb_spec.packing_algorithm)` |
| `areal/infra/dist_rollout.py` | Added `packing_algorithm` parameter to `redistribute_trajectories()` |
| `tests/test_kk_allocate.py` | Unit tests for KK correctness, config dispatch, FFD comparison |
| `tests/test_kk_e2e.py` | End-to-end 4-GPU test validating KK advantage |
| `tests/torchrun/run_kk_vs_ffd.py` | Torchrun worker for distributed KK vs FFD comparison |
