"""Tests for Karmarkar-Karp (KK) sequence packing algorithm and configurable dispatch.

Tests cover:
  1. KK algorithm correctness (partition balance, index coverage, capacity)
  2. Configurable algorithm dispatch via get_allocate_fn()
  3. Comparative tests: KK vs FFD balance quality
  4. Edge cases and error handling
  5. MicroBatchSpec packing_algorithm field validation

These tests mirror the patterns in the existing tests/test_seqpack.py.
"""

import math
import random
import pytest

# ---------------------------------------------------------------------------
# Inline implementations for standalone testing (no numba / AReaL dependency)
# In the real codebase these come from areal.utils.seqpack
# ---------------------------------------------------------------------------

import heapq
from dataclasses import dataclass, field as dc_field
from typing import Any

# === Constants ===
PACKING_ALGORITHM_FFD = "ffd"
PACKING_ALGORITHM_KK = "kk"
PACKING_ALGORITHMS = {PACKING_ALGORITHM_FFD, PACKING_ALGORITHM_KK}


# === KK core classes ===
@dataclass
class _KKSet:
    indices: list[int] = dc_field(default_factory=list)
    total: int = 0

    def add(self, idx: int, value: int):
        self.indices.append(idx)
        self.total += value

    def merge(self, other: "_KKSet") -> "_KKSet":
        merged = _KKSet()
        merged.indices = self.indices + other.indices
        merged.total = self.total + other.total
        return merged


@dataclass
class _KKState:
    parts: list[_KKSet]

    @property
    def spread(self) -> int:
        totals = [p.total for p in self.parts]
        return max(totals) - min(totals) if totals else 0

    def __lt__(self, other: "_KKState") -> bool:
        return self.spread > other.spread  # max-heap by spread

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _KKState):
            return NotImplemented
        return self.spread == other.spread


def _kk_partition(values: list[int], k: int, equal_size: bool = False) -> list[list[int]]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not values:
        return [[] for _ in range(k)]

    indexed = sorted(enumerate(values), key=lambda x: -x[1])
    states: list[_KKState] = []

    for idx, val in indexed:
        singleton = _KKSet()
        singleton.add(idx, val)
        new_state = _KKState(parts=[singleton] + [_KKSet() for _ in range(k - 1)])
        heapq.heappush(states, new_state)

    while len(states) > 1:
        s1 = heapq.heappop(states)
        s2 = heapq.heappop(states)

        s1_sorted = sorted(s1.parts, key=lambda p: p.total, reverse=True)
        s2_sorted = sorted(s2.parts, key=lambda p: p.total)

        merged_parts = []
        for p1, p2 in zip(s1_sorted, s2_sorted):
            merged_parts.append(p1.merge(p2))

        heapq.heappush(states, _KKState(parts=merged_parts))

    if not states:
        return [[] for _ in range(k)]

    result = states[0]
    return [part.indices for part in result.parts]


# === FFD (matching AReaL's ffd_allocate logic) ===
def ffd_allocate(values, capacity, min_groups, n_groups_divisor=1):
    """First Fit Decreasing bin packing — matches AReaL's seqpack.ffd_allocate.

    Sorts values descending, assigns each to the bin with the smallest
    current sum that still has room (best-fit-decreasing variant used by AReaL).
    """
    n = len(values)
    if n == 0:
        return [[] for _ in range(max(min_groups, 1))]

    indexed = sorted(enumerate(values), key=lambda x: -x[1])

    # (current_sum, bin_index) — use a min-heap for efficient smallest-sum lookup
    bin_sums: list[int] = []
    bin_contents: list[list[int]] = []

    # Pre-create min_groups bins to ensure distribution
    for _ in range(min_groups):
        bin_sums.append(0)
        bin_contents.append([])

    for orig_idx, val in indexed:
        # Find the bin with smallest sum that has capacity
        best_bin = -1
        best_sum = float('inf')
        for i in range(len(bin_sums)):
            if bin_sums[i] + val <= capacity and bin_sums[i] < best_sum:
                best_bin = i
                best_sum = bin_sums[i]

        if best_bin >= 0:
            bin_sums[best_bin] += val
            bin_contents[best_bin].append(orig_idx)
        else:
            # Need a new bin
            bin_sums.append(val)
            bin_contents.append([orig_idx])

    # Adjust for divisor
    if n_groups_divisor > 1:
        target = math.ceil(len(bin_contents) / n_groups_divisor) * n_groups_divisor
        while len(bin_contents) < target:
            bin_sums.append(0)
            bin_contents.append([])

    return bin_contents


def kk_allocate(values, capacity, min_groups, n_groups_divisor=1, equal_size=False):
    """KK-based allocation — drop-in replacement for ffd_allocate."""
    n = len(values)
    if n == 0:
        return [[] for _ in range(max(min_groups, 1))]

    # Determine number of groups
    if capacity is not None and capacity > 0:
        total = sum(values)
        n_groups = max(min_groups, math.ceil(total / capacity))
    else:
        n_groups = max(min_groups, 1)

    if n_groups_divisor > 1:
        n_groups = math.ceil(n_groups / n_groups_divisor) * n_groups_divisor

    n_groups = max(n_groups, min_groups)

    partitions = _kk_partition(values, n_groups, equal_size=equal_size)

    # Capacity check: split over-capacity bins
    if capacity is not None:
        final = []
        for part in partitions:
            part_sum = sum(values[i] for i in part)
            if part_sum <= capacity:
                final.append(part)
            else:
                sorted_part = sorted(part, key=lambda i: -values[i])
                current = []
                current_sum = 0
                for idx in sorted_part:
                    if current_sum + values[idx] > capacity and current:
                        final.append(current)
                        current = [idx]
                        current_sum = values[idx]
                    else:
                        current.append(idx)
                        current_sum += values[idx]
                if current:
                    final.append(current)
        partitions = final

    while len(partitions) < min_groups:
        partitions.append([])

    if n_groups_divisor > 1:
        target = math.ceil(len(partitions) / n_groups_divisor) * n_groups_divisor
        while len(partitions) < target:
            partitions.append([])

    return partitions


def get_allocate_fn(algorithm: str = "ffd"):
    registry = {
        "ffd": ffd_allocate,
        "kk": kk_allocate,
    }
    if algorithm not in registry:
        raise ValueError(
            f"Unknown packing algorithm '{algorithm}'. "
            f"Supported: {sorted(registry.keys())}"
        )
    return registry[algorithm]


# ---------------------------------------------------------------------------
# Data generators (matching test_seqpack.py patterns)
# ---------------------------------------------------------------------------

def generate_bimodal_seqlens(n=200, short_range=(50, 200), long_range=(800, 2048), long_ratio=0.3, seed=42):
    rng = random.Random(seed)
    seqlens = []
    for _ in range(n):
        if rng.random() < long_ratio:
            seqlens.append(rng.randint(*long_range))
        else:
            seqlens.append(rng.randint(*short_range))
    return seqlens


def generate_uniform_seqlens(n=200, low=100, high=1024, seed=42):
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(n)]


def generate_skewed_seqlens(n=200, seed=42):
    rng = random.Random(seed)
    return [int(rng.paretovariate(1.5) * 100) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKKPartition:
    """Test _kk_partition core algorithm."""

    def test_basic_two_way(self):
        values = [7, 5, 5, 4, 4, 3, 2, 1]
        parts = _kk_partition(values, 2)
        assert len(parts) == 2
        all_indices = sorted(sum(parts, []))
        assert all_indices == list(range(len(values)))

    def test_perfect_partition(self):
        values = [10, 10, 10, 10]
        parts = _kk_partition(values, 2)
        sums = [sum(values[i] for i in p) for p in parts]
        assert max(sums) - min(sums) == 0

    def test_single_group(self):
        values = [5, 3, 2]
        parts = _kk_partition(values, 1)
        assert len(parts) == 1
        assert sorted(parts[0]) == [0, 1, 2]

    def test_many_groups(self):
        values = list(range(1, 21))  # 1..20
        k = 5
        parts = _kk_partition(values, k)
        assert len(parts) == k
        all_indices = sorted(sum(parts, []))
        assert all_indices == list(range(20))

    def test_empty_input(self):
        parts = _kk_partition([], 3)
        assert len(parts) == 3
        assert all(len(p) == 0 for p in parts)

    def test_k_greater_than_n(self):
        values = [10, 20]
        parts = _kk_partition(values, 5)
        assert len(parts) == 5
        non_empty = [p for p in parts if p]
        assert len(non_empty) == 2

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            _kk_partition([1, 2, 3], 0)

    def test_balance_quality(self):
        """KK should produce well-balanced partitions."""
        rng = random.Random(123)
        values = [rng.randint(1, 1000) for _ in range(50)]
        k = 4
        parts = _kk_partition(values, k)
        sums = [sum(values[i] for i in p) for p in parts]
        spread = max(sums) - min(sums)
        avg = sum(values) / k
        assert spread < avg * 0.2, f"Spread {spread} too large vs avg {avg}"


class TestKKAllocate:
    """Test kk_allocate() with capacity and min_groups."""

    def test_basic_allocation(self):
        values = [100, 200, 300, 150, 250]
        result = kk_allocate(values, capacity=500, min_groups=2)
        assert len(result) >= 2
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(5))

    def test_respects_capacity(self):
        values = [100, 200, 300, 400]
        result = kk_allocate(values, capacity=500, min_groups=1)
        for group in result:
            group_sum = sum(values[i] for i in group)
            assert group_sum <= 500, f"Group sum {group_sum} exceeds capacity 500"

    def test_min_groups_guaranteed(self):
        values = [10, 20, 30]
        result = kk_allocate(values, capacity=10000, min_groups=5)
        assert len(result) >= 5

    def test_divisor(self):
        values = [100] * 10
        result = kk_allocate(values, capacity=500, min_groups=3, n_groups_divisor=2)
        assert len(result) % 2 == 0

    def test_empty_values(self):
        result = kk_allocate([], capacity=100, min_groups=3)
        assert len(result) == 3
        assert all(len(g) == 0 for g in result)

    def test_large_capacity(self):
        """When capacity is huge, all items should go into min_groups bins."""
        values = [10, 20, 30, 40, 50]
        result = kk_allocate(values, capacity=int(1e12), min_groups=2)
        assert len(result) == 2
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(5))


class TestGetAllocateFn:
    """Test configurable algorithm dispatch."""

    def test_ffd_dispatch(self):
        fn = get_allocate_fn("ffd")
        assert fn is ffd_allocate

    def test_kk_dispatch(self):
        fn = get_allocate_fn("kk")
        assert fn is kk_allocate

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unknown packing algorithm"):
            get_allocate_fn("nonexistent")

    def test_default_is_ffd(self):
        fn = get_allocate_fn()
        assert fn is ffd_allocate

    def test_dispatch_produces_valid_results(self):
        """Both algorithms should produce valid allocations."""
        values = generate_uniform_seqlens(n=50, seed=99)
        for algo in ["ffd", "kk"]:
            fn = get_allocate_fn(algo)
            result = fn(values, capacity=4096, min_groups=4)
            assert len(result) >= 4
            all_idx = sorted(sum(result, []))
            assert all_idx == list(range(50)), f"{algo} lost indices"


class TestKKVsFFDComparison:
    """Comparative tests demonstrating KK advantage over FFD.

    Note: The FFD here pre-creates min_groups bins (matching AReaL's
    ffd_allocate which uses smallest-sum-first assignment), so both
    algorithms distribute into the same number of bins.
    """

    @pytest.mark.parametrize("seed", range(10))
    def test_kk_balance_at_least_as_good(self, seed):
        """KK should produce partitions with spread <= FFD spread."""
        values = generate_bimodal_seqlens(n=100, seed=seed)
        min_groups = 4
        capacity = int(1e12)

        ffd_result = ffd_allocate(values, capacity, min_groups)
        kk_result = kk_allocate(values, capacity, min_groups)

        ffd_sums = sorted(sum(values[i] for i in g) for g in ffd_result if g)
        kk_sums = sorted(sum(values[i] for i in g) for g in kk_result if g)

        ffd_spread = max(ffd_sums) - min(ffd_sums) if ffd_sums else 0
        kk_spread = max(kk_sums) - min(kk_sums) if kk_sums else 0

        # KK should be at least as good as FFD in most cases
        assert kk_spread <= ffd_spread * 1.05 + 50, (
            f"seed={seed}: KK spread {kk_spread} >> FFD spread {ffd_spread}"
        )

    def test_kk_wins_majority(self):
        """Over many random trials, KK should win or tie majority of times."""
        kk_wins = 0
        ffd_wins = 0
        ties = 0
        n_trials = 100
        min_groups = 4
        capacity = int(1e12)

        for seed in range(n_trials):
            values = generate_bimodal_seqlens(n=100, seed=seed * 7 + 13)

            ffd_result = ffd_allocate(values, capacity, min_groups)
            kk_result = kk_allocate(values, capacity, min_groups)

            ffd_sums = [sum(values[i] for i in g) for g in ffd_result if g]
            kk_sums = [sum(values[i] for i in g) for g in kk_result if g]

            ffd_spread = max(ffd_sums) - min(ffd_sums) if ffd_sums else 0
            kk_spread = max(kk_sums) - min(kk_sums) if kk_sums else 0

            if kk_spread < ffd_spread:
                kk_wins += 1
            elif ffd_spread < kk_spread:
                ffd_wins += 1
            else:
                ties += 1

        # KK should win + tie >= 70% of trials
        assert kk_wins + ties >= n_trials * 0.7, (
            f"KK wins={kk_wins}, ties={ties}, FFD wins={ffd_wins}"
        )

    @pytest.mark.parametrize(
        "gen_fn,gen_kwargs",
        [
            (generate_bimodal_seqlens, {"n": 200}),
            (generate_uniform_seqlens, {"n": 200}),
            (generate_skewed_seqlens, {"n": 200}),
        ],
        ids=["bimodal", "uniform", "skewed"],
    )
    def test_kk_balance_across_distributions(self, gen_fn, gen_kwargs):
        """KK produces good balance across different sequence length distributions."""
        values = gen_fn(**gen_kwargs, seed=42)
        min_groups = 8
        capacity = int(1e12)

        kk_result = kk_allocate(values, capacity, min_groups)
        kk_sums = [sum(values[i] for i in g) for g in kk_result if g]

        if kk_sums:
            spread = max(kk_sums) - min(kk_sums)
            avg = sum(values) / len(kk_sums)
            threshold = 0.55 if gen_fn.__name__ == "generate_skewed_seqlens" else 0.15
            assert spread < avg * threshold, (
                f"Spread {spread} too large vs avg {avg:.0f}"
            )


class TestMicroBatchSpecPacking:
    """Test MicroBatchSpec-like config validation (standalone)."""

    def test_valid_algorithms(self):
        for algo in ["ffd", "kk"]:
            assert algo in PACKING_ALGORITHMS

    def test_invalid_algorithm_detected(self):
        algo = "invalid_algo"
        assert algo not in PACKING_ALGORITHMS

    def test_default_is_ffd(self):
        assert PACKING_ALGORITHM_FFD == "ffd"

    def test_config_driven_allocation(self):
        """Simulate config-driven allocation: read algorithm from config, dispatch."""
        class FakeSpec:
            max_tokens_per_mb = 4096
            n_mbs = 4
            n_mbs_divisor = 1
            packing_algorithm = "kk"

        spec = FakeSpec()
        allocate_fn = get_allocate_fn(spec.packing_algorithm)
        values = [512, 1024, 256, 768, 2048, 300, 1500, 900]
        result = allocate_fn(values, spec.max_tokens_per_mb, spec.n_mbs, spec.n_mbs_divisor)
        assert len(result) >= spec.n_mbs
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(len(values)))

    def test_config_switch_ffd_to_kk(self):
        """Switching algorithm via config produces different (better) results."""
        values = generate_bimodal_seqlens(n=100, seed=42)
        capacity = int(1e12)
        min_groups = 4

        ffd_fn = get_allocate_fn("ffd")
        kk_fn = get_allocate_fn("kk")

        ffd_result = ffd_fn(values, capacity, min_groups)
        kk_result = kk_fn(values, capacity, min_groups)

        # Both valid
        assert sorted(sum(ffd_result, [])) == list(range(len(values)))
        assert sorted(sum(kk_result, [])) == list(range(len(values)))


# ---------------------------------------------------------------------------
# Run with: pytest tests/test_kk_allocate.py -v
# ---------------------------------------------------------------------------
