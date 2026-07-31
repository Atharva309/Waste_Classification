"""
Systematic inference latency benchmarking.

Fixes issue #7: the old project reported one incidental ~20ms number from
YOLOv5's own console output, with no methodology (no warmup, no repeats,
no percentile reporting, unclear whether it was batch=1 or batched).

For a moving-belt / moving-camera real-time system, what matters operationally
is the tail latency of single-frame (batch=1) inference, since frames arrive
one at a time and a slow outlier frame is a missed or late spray decision --
mean latency alone hides that. This module:
  - runs N warmup iterations (excluded from stats -- first-call overhead,
    lazy CUDA/cuDNN init, etc. would otherwise skew results)
  - times M measured iterations individually (not just total/M, since we
    want the distribution, not just the mean)
  - reports mean, std, p50, p90, p95, p99, max
  - is model-framework-agnostic: `predict_fn` is just a callable, so the
    same benchmark harness works for the Keras models, the PyTorch models,
    or the whole detect+NMS pipeline end-to-end.
"""
from __future__ import annotations
import time
import statistics
from dataclasses import dataclass, asdict


@dataclass
class LatencyStats:
    n_iters: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    throughput_fps: float

    def as_dict(self):
        return asdict(self)


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def benchmark_latency(predict_fn, make_input_fn, n_warmup=10, n_iters=100) -> LatencyStats:
    """predict_fn(x) -> runs one inference call.
    make_input_fn() -> produces one fresh input each call (batch=1 realistic
    for a belt scenario -- items arrive one at a time, not in convenient
    fixed-size batches).
    """
    for _ in range(n_warmup):
        predict_fn(make_input_fn())

    times_ms = []
    for _ in range(n_iters):
        x = make_input_fn()
        t0 = time.perf_counter()
        predict_fn(x)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    mean = statistics.mean(times_ms)
    std = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    return LatencyStats(
        n_iters=n_iters,
        mean_ms=mean, std_ms=std,
        p50_ms=_percentile(times_ms, 50), p90_ms=_percentile(times_ms, 90),
        p95_ms=_percentile(times_ms, 95), p99_ms=_percentile(times_ms, 99),
        max_ms=max(times_ms), min_ms=min(times_ms),
        throughput_fps=1000.0 / mean if mean > 0 else 0.0,
    )
