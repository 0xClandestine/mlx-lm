"""phew benchmark — RMSNorm and SDPA on DeepSeek-V4 geometry.

Shows speedup from phew's primitive substitution pass:
  - RMSNorm: manual x/sqrt(mean(x²)+eps)*w  → mx.fast.rms_norm       (~2.8×)
  - SDPA:    manual softmax(Q@K^T*s)@V       → mx.fast.scaled_dot_product_attention (~1.4×)

Dims: hidden=4096 (DeepSeek-V4), SDPA uses H=32 D=128 standard geometry.

Usage:
    pip install phew-mlx
    python benchmarks/bench_phew.py
"""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

# DeepSeek-V4 hidden_size; SDPA uses standard H=32 D=128 geometry
HIDDEN          = 4096
SDPA_B, SDPA_S  = 4, 512
SDPA_H, SDPA_D  = 32, 128


def bench(fn, *args, warmup: int = 20, n: int = 100) -> float:
    for _ in range(warmup):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# ---------------------------------------------------------------------------
# RMSNorm — manual vs mx.fast.rms_norm
# ---------------------------------------------------------------------------
def bench_rms_norm():
    from phew import Optimizer
    from phew.verify import SubstitutionClass

    w = mx.random.normal((HIDDEN,)); mx.eval(w)

    def rms_norm(x, w):
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(ms + 1e-5) * w

    x0 = mx.random.normal((SDPA_B, SDPA_S, HIDDEN)); mx.eval(x0)
    baseline_ms = bench(rms_norm, x0, w)

    def input_factory(tag, seed):
        mx.random.seed(seed)
        return ([mx.random.normal((SDPA_B, SDPA_S, HIDDEN)), w], {})

    opt = Optimizer(fn=rms_norm, input_factory=input_factory,
        enabled_subst_classes={SubstitutionClass.fp32_to_fp32}, fn_name="rms_norm")
    result = opt.run()
    return baseline_ms, result.optimized_ms, result.applied_rules, result.output_source


# ---------------------------------------------------------------------------
# SDPA — manual softmax attention vs mx.fast.scaled_dot_product_attention
# ---------------------------------------------------------------------------
def bench_sdpa():
    from phew import Optimizer
    from phew.verify import SubstitutionClass

    scale = SDPA_D ** -0.5

    def sdpa(q, k, v):
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        weights = mx.softmax(scores, axis=-1)
        return weights @ v

    q = mx.random.normal((SDPA_B, SDPA_H, SDPA_S, SDPA_D)); mx.eval(q)
    k = mx.random.normal((SDPA_B, SDPA_H, SDPA_S, SDPA_D)); mx.eval(k)
    v = mx.random.normal((SDPA_B, SDPA_H, SDPA_S, SDPA_D)); mx.eval(v)
    baseline_ms = bench(sdpa, q, k, v)

    def input_factory(tag, seed):
        mx.random.seed(seed)
        return ([mx.random.normal((SDPA_B, SDPA_H, SDPA_S, SDPA_D))] * 3, {})

    opt = Optimizer(fn=sdpa, input_factory=input_factory,
        enabled_subst_classes={SubstitutionClass.fp32_to_fp32}, fn_name="sdpa")
    result = opt.run()
    return baseline_ms, result.optimized_ms, result.applied_rules, result.output_source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not mx.metal.is_available():
        print("WARNING: Metal not available — results will not reflect GPU performance.")

    print(f"\nphew × DeepSeek-V4  (pip install phew-mlx)")
    print(f"{'kernel':<40}  {'baseline':>9}  {'optimised':>9}  {'speedup':>7}  passes")
    print("-" * 82)

    for label, fn in [
        (f"RMSNorm  (B={SDPA_B} S={SDPA_S} H={HIDDEN})",           bench_rms_norm),
        (f"SDPA     (B={SDPA_B} S={SDPA_S} H={SDPA_H} D={SDPA_D})", bench_sdpa),
    ]:
        try:
            bl, opt_ms, passes, src = fn()
            speedup = bl / opt_ms
            sig = "  *" if speedup > 1.03 else ""
            passes_str = ", ".join(passes) if passes else "none"
            print(
                f"{label:<40}  {bl:>9.3f}ms  {opt_ms:>9.3f}ms  {speedup:>6.2f}×{sig}"
                f"  [{passes_str}]"
            )
            print(f"  emitted: {src.splitlines()[3].strip()}")
        except Exception as exc:
            import traceback
            print(f"{label:<40}  ERROR: {exc}")
            traceback.print_exc()

    print("-" * 82)
    print("* = speedup outside ±3% noise band\n")
