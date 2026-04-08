"""Torchrun worker script: compare KK vs FFD for trajectory redistribution.

Launched by test_kk_e2e.py via torchrun. Each rank:
  1. Creates synthetic trajectories with bimodal sequence lengths
  2. Runs redistribute_trajectories with FFD
  3. Runs redistribute_trajectories with KK
  4. Writes per-rank metrics (total tokens, spread) to pickle files

Usage (launched by pytest, not directly):
  torchrun --nproc_per_node=4 tests/torchrun/run_kk_vs_ffd.py \
      --output_dir /tmp/kk_test --n_seqs 200 --seed 42
"""

import argparse
import os
import pickle
import random
import sys
import math
import heapq
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


# =====================================================================
# Inline KK + FFD implementations (self-contained, no AReaL imports)
# =====================================================================

@dataclass
class _KKSet:
    indices: list[int] = field(default_factory=list)
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
        return self.spread > other.spread

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _KKState):
            return NotImplemented
        return self.spread == other.spread


def _kk_partition(values, k):
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not values:
        return [[] for _ in range(k)]

    indexed = sorted(enumerate(values), key=lambda x: -x[1])
    states = []

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
        merged_parts = [p1.merge(p2) for p1, p2 in zip(s1_sorted, s2_sorted)]
        heapq.heappush(states, _KKState(parts=merged_parts))

    return [part.indices for part in states[0].parts] if states else [[] for _ in range(k)]


def kk_allocate(values, capacity, min_groups, n_groups_divisor=1):
    n = len(values)
    if n == 0:
        return [[] for _ in range(max(min_groups, 1))]

    if capacity is not None and capacity > 0:
        total = sum(values)
        n_groups = max(min_groups, math.ceil(total / capacity))
    else:
        n_groups = max(min_groups, 1)

    if n_groups_divisor > 1:
        n_groups = math.ceil(n_groups / n_groups_divisor) * n_groups_divisor

    partitions = _kk_partition(values, n_groups)

    if capacity is not None:
        final = []
        for part in partitions:
            part_sum = sum(values[i] for i in part)
            if part_sum <= capacity:
                final.append(part)
            else:
                sorted_part = sorted(part, key=lambda i: -values[i])
                current, current_sum = [], 0
                for idx in sorted_part:
                    if current_sum + values[idx] > capacity and current:
                        final.append(current)
                        current, current_sum = [idx], values[idx]
                    else:
                        current.append(idx)
                        current_sum += values[idx]
                if current:
                    final.append(current)
        partitions = final

    while len(partitions) < min_groups:
        partitions.append([])
    return partitions


def ffd_allocate(values, capacity, min_groups, n_groups_divisor=1):
    n = len(values)
    if n == 0:
        return [[] for _ in range(max(min_groups, 1))]

    indexed = sorted(enumerate(values), key=lambda x: -x[1])
    bins = []

    for idx, val in indexed:
        placed = False
        for i, (s, indices) in enumerate(bins):
            if s + val <= capacity:
                bins[i] = (s + val, indices + [idx])
                placed = True
                break
        if not placed:
            bins.append((val, [idx]))

    while len(bins) < min_groups:
        bins.append((0, []))

    if n_groups_divisor > 1:
        target = math.ceil(len(bins) / n_groups_divisor) * n_groups_divisor
        while len(bins) < target:
            bins.append((0, []))

    return [indices for _, indices in bins]


# =====================================================================
# Simulated redistribute_trajectories (mirrors dist_rollout.py logic)
# =====================================================================

def redistribute_trajectories_sim(seqlens, world_size, algorithm="ffd"):
    """Simulate trajectory redistribution: allocate seqlens to ranks."""
    allocate_fn = kk_allocate if algorithm == "kk" else ffd_allocate
    groups = allocate_fn(seqlens, capacity=int(1e12), min_groups=world_size)

    # Assign groups to ranks round-robin (simplified)
    rank_loads = [0] * world_size
    rank_indices = [[] for _ in range(world_size)]

    for group in groups:
        # Assign to least-loaded rank
        min_rank = min(range(world_size), key=lambda r: rank_loads[r])
        rank_indices[min_rank].extend(group)
        rank_loads[min_rank] += sum(seqlens[i] for i in group)

    return rank_indices, rank_loads


# =====================================================================
# Main worker
# =====================================================================

def generate_bimodal_seqlens(n, short_range=(50, 200), long_range=(800, 2048), long_ratio=0.3, seed=42):
    rng = random.Random(seed)
    seqlens = []
    for _ in range(n):
        if rng.random() < long_ratio:
            seqlens.append(rng.randint(*long_range))
        else:
            seqlens.append(rng.randint(*short_range))
    return seqlens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_seqs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    os.makedirs(args.output_dir, exist_ok=True)

    # All ranks use the same data
    seqlens = generate_bimodal_seqlens(args.n_seqs, seed=args.seed)

    # Run FFD redistribution
    ffd_indices, ffd_loads = redistribute_trajectories_sim(seqlens, world_size, "ffd")

    # Run KK redistribution
    kk_indices, kk_loads = redistribute_trajectories_sim(seqlens, world_size, "kk")

    # Compute metrics
    ffd_spread = max(ffd_loads) - min(ffd_loads)
    kk_spread = max(kk_loads) - min(kk_loads)
    ffd_max = max(ffd_loads)
    kk_max = max(kk_loads)

    result = {
        "rank": rank,
        "world_size": world_size,
        "n_seqs": args.n_seqs,
        "total_tokens": sum(seqlens),
        "ffd_loads": ffd_loads,
        "kk_loads": kk_loads,
        "ffd_spread": ffd_spread,
        "kk_spread": kk_spread,
        "ffd_max_load": ffd_max,
        "kk_max_load": kk_max,
        "kk_wins": kk_spread < ffd_spread,
        "improvement_pct": (
            (ffd_spread - kk_spread) / ffd_spread * 100 if ffd_spread > 0 else 0.0
        ),
    }

    output_path = os.path.join(args.output_dir, f"rank_{rank}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    # Synchronize
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
