import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.latency import benchmark_latency, _percentile

def test_percentile_known_values():
    vals = [1,2,3,4,5,6,7,8,9,10]
    assert abs(_percentile(vals, 50) - 5.5) < 1e-6
    assert _percentile(vals, 0) == 1
    assert _percentile(vals, 100) == 10

def test_benchmark_latency_reports_sane_stats():
    def predict_fn(x):
        time.sleep(0.001)  # simulate ~1ms inference
        return x

    stats = benchmark_latency(predict_fn, make_input_fn=lambda: random.random(),
                               n_warmup=3, n_iters=20)
    assert stats.n_iters == 20
    assert 0.5 < stats.mean_ms < 20  # sleep(0.001) plus overhead, generous bounds for CI noise
    assert stats.p50_ms <= stats.p90_ms <= stats.p95_ms <= stats.p99_ms <= stats.max_ms
    assert stats.min_ms <= stats.p50_ms
    assert stats.throughput_fps > 0

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"\n{len(fns)} latency tests passed")
