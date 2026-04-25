# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import RotatingKVCache
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU


def _default_quantization() -> Dict:
    return {"group_size": 32, "bits": 4, "mode": "mxfp4"}


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False
    topk_method: str = "noaux_tc"
    quantization: Optional[Dict] = field(default_factory=_default_quantization)
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")
        if self.quantization is None:
            self.quantization = _default_quantization()


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(mx.logaddexp(scores, mx.zeros_like(scores)))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
    ):
        super().__init__()
        self.dims = dims

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
        inverse: bool = False,
        positions: Optional[mx.array] = None,
    ):
        dtype = x.dtype
        L = x.shape[-2]
        pos = (
            mx.arange(offset, offset + L, dtype=mx.float32)
            if positions is None
            else positions.astype(mx.float32)
        )
        freqs = pos[:, None] * self.inv_freq[None, :]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        if inverse:
            sin = -sin

        broadcast_shape = (1,) * (x.ndim - 2) + cos.shape
        cos = cos.reshape(broadcast_shape).astype(dtype)
        sin = sin.reshape(broadcast_shape).astype(dtype)

        x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
        return out.reshape(*out.shape[:-2], out.shape[-2] * 2)


def _make_partial_rope_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;  // 0..31 (one SIMD group)
        uint gid = threadgroup_position_in_grid.x;    // one per (b, h, l) triplet

        constexpr int D = NOPE + 2 * DRH;
        int L_v = dims[0];
        int H_v = dims[1];
        uint l   = gid % (uint)L_v;
        uint tmp = gid / (uint)L_v;
        uint h   = tmp % (uint)H_v;
        uint b   = tmp / (uint)H_v;

        const auto xp = x     + ((uint64_t)b * H_v * L_v + h * L_v + l) * D;
        auto       yp = y     + ((uint64_t)b * H_v * L_v + h * L_v + l) * D;
        const auto cp = cos_s + l * DRH;
        const auto sp = sin_s + l * DRH;

        // Copy the nope prefix (stride-32 coalesced across the SIMD group)
        for (int i = (int)tid; i < NOPE; i += 32)
            store_elem(yp[i], float(xp[i]));

        // Apply RoPE rotation to the trailing D_ROPE = 2*DRH elements.
        // Pairs are interleaved: (x[2i], x[2i+1]) rotated by freq i.
        // Each of the 32 lanes handles one interleaved pair.
        if ((int)tid < DRH) {
            float x0 = float(xp[NOPE + 2 * tid]);
            float x1 = float(xp[NOPE + 2 * tid + 1]);
            float c  = float(cp[tid]);
            float s  = float(sp[tid]);
            if (INVERSE) {
                store_elem(yp[NOPE + 2 * tid],     fma( x1, s, x0 * c));   //  x0*c + x1*s
                store_elem(yp[NOPE + 2 * tid + 1], fma(-x0, s, x1 * c));   // -x0*s + x1*c
            } else {
                store_elem(yp[NOPE + 2 * tid],     fma(-x1, s, x0 * c));   //  x0*c - x1*s
                store_elem(yp[NOPE + 2 * tid + 1], fma( x0, s, x1 * c));   //  x0*s + x1*c
            }
        }
    """
    return mx.fast.metal_kernel(
        name="ds4_partial_rope",
        input_names=["x", "cos_s", "sin_s", "dims"],
        output_names=["y"],
        header="template<typename T> inline void store_elem(device T& dst, float v) { dst = T(v); }",
        source=source,
    )


_partial_rope_kernel = _make_partial_rope_kernel()


def _make_q_norm_kernel():
    """Per-head RMS norm for query vectors: fuses sq-accumulate + rsqrt + scale in one pass."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;  // 0..31 (one SIMD group)
        uint gid = threadgroup_position_in_grid.x;    // one per (b, l, h) triplet

        int D   = dims[0];
        int L_v = dims[1];
        int H_v = dims[2];
        float eps_v = float(eps[0]);

        uint h   = gid % (uint)H_v;
        uint tmp = gid / (uint)H_v;
        uint l   = tmp % (uint)L_v;
        uint b   = tmp / (uint)L_v;

        const auto xp = x + ((uint64_t)b * L_v * H_v + l * H_v + h) * D;
        auto       yp = y + ((uint64_t)b * L_v * H_v + l * H_v + h) * D;

        // Pass 1: accumulate partial sum of squares (stride-32 coalesced)
        float partial_sq = 0.f;
        for (int i = (int)tid; i < D; i += 32)
            partial_sq = fma(float(xp[i]), float(xp[i]), partial_sq);

        float rms_scale = metal::fast::rsqrt(simd_sum(partial_sq) / float(D) + eps_v);

        // Pass 2: apply scale (D=512 fits in L1 cache; second pass is cache-warm)
        for (int i = (int)tid; i < D; i += 32)
            store_elem(yp[i], float(xp[i]) * rms_scale);
    """
    return mx.fast.metal_kernel(
        name="ds4_q_norm",
        input_names=["x", "eps", "dims"],
        output_names=["y"],
        header="template<typename T> inline void store_elem(device T& dst, float v) { dst = T(v); }",
        source=source,
    )


_q_norm_kernel = _make_q_norm_kernel()


def _apply_partial_rope(
    x: mx.array,
    rope: "DeepseekV4RoPE",
    offset: int = 0,
    inverse: bool = False,
    positions: Optional[mx.array] = None,
) -> mx.array:
    rope_dim = rope.dims
    nope_dim = x.shape[-1] - rope_dim
    L = x.shape[-2]

    if _partial_rope_kernel is not None:
        B, H = x.shape[0], x.shape[1]
        pos = (
            mx.arange(offset, offset + L, dtype=mx.float32)
            if positions is None
            else positions.astype(mx.float32)
        )
        freqs = pos[:, None] * rope.inv_freq[None, :]
        cos = mx.cos(freqs).astype(mx.float32)
        sin = mx.sin(freqs).astype(mx.float32)
        dims_arr = mx.array([L, H], dtype=mx.int32)
        return _partial_rope_kernel(
            inputs=[x, cos, sin, dims_arr],
            template=[
                ("NOPE", nope_dim),
                ("DRH", rope_dim // 2),
                ("INVERSE", 1 if inverse else 0),
            ],
            grid=(B * H * L * 32, 1, 1),  # grid = total threads; 32 threads per (b,h,l)
            threadgroup=(32, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]

    # Fallback: original slice-rotate-concat path
    if nope_dim == 0:
        return rope(x, offset=offset, inverse=inverse, positions=positions)
    nope, pe = mx.split(x, [nope_dim], axis=-1)
    pe = rope(pe, offset=offset, inverse=inverse, positions=positions)
    return mx.concatenate([nope, pe], axis=-1)


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _make_hc_split_sinkhorn_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint idx = thread_position_in_grid.x;
        if (idx >= (uint)n_rows[0]) return;
        constexpr int MIX = (2 + HC) * HC;
        float epsv = static_cast<float>(eps[0]);

        const auto mix = mixes + idx * MIX;
        auto pre_out   = pre   + idx * HC;
        auto post_out  = post  + idx * HC;
        auto comb_out  = comb  + idx * HC * HC;

        const float pre_scale  = static_cast<float>(scale[0]);
        const float post_scale = static_cast<float>(scale[1]);
        const float comb_scale = static_cast<float>(scale[2]);

        // Pre-sigmoid: float4 vectorized (HC == 4)
        {
            float4 z = float4(
                static_cast<float>(mix[0]), static_cast<float>(mix[1]),
                static_cast<float>(mix[2]), static_cast<float>(mix[3])
            ) * pre_scale + float4(
                static_cast<float>(base[0]), static_cast<float>(base[1]),
                static_cast<float>(base[2]), static_cast<float>(base[3])
            );
            *(device float4*)pre_out = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }

        // Post-sigmoid: float4 vectorized
        {
            float4 z = float4(
                static_cast<float>(mix[4]), static_cast<float>(mix[5]),
                static_cast<float>(mix[6]), static_cast<float>(mix[7])
            ) * post_scale + float4(
                static_cast<float>(base[4]), static_cast<float>(base[5]),
                static_cast<float>(base[6]), static_cast<float>(base[7])
            );
            *(device float4*)post_out = 2.0f / (1.0f + metal::fast::exp(-z));
        }

        // Comb: 4x4 matrix as float4 rows; tree-reduce max, vectorized exp + dot-product sum
        constexpr int BASE = 2 * HC;
        float4 rows[4];
        for (int i = 0; i < 4; ++i) {
            float4 v = float4(
                fma(static_cast<float>(mix[BASE + i*HC + 0]), comb_scale, static_cast<float>(base[BASE + i*HC + 0])),
                fma(static_cast<float>(mix[BASE + i*HC + 1]), comb_scale, static_cast<float>(base[BASE + i*HC + 1])),
                fma(static_cast<float>(mix[BASE + i*HC + 2]), comb_scale, static_cast<float>(base[BASE + i*HC + 2])),
                fma(static_cast<float>(mix[BASE + i*HC + 3]), comb_scale, static_cast<float>(base[BASE + i*HC + 3]))
            );
            float m = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - m);
            rows[i] = e * 1.0f /(dot(e, float4(1.0f))) + epsv;
        }

        // Initial column normalization
        {
            float4 inv_c = 1.0f /(rows[0] + rows[1] + rows[2] + rows[3] + epsv);
            rows[0] *= inv_c; rows[1] *= inv_c;
            rows[2] *= inv_c; rows[3] *= inv_c;
        }

        // Sinkhorn iterations: row-normalize then column-normalize
        for (int iter = 1; iter < ITERS; ++iter) {
            for (int i = 0; i < 4; ++i)
                rows[i] *= 1.0f /(dot(rows[i], float4(1.0f)) + epsv);
            float4 inv_c = 1.0f /(rows[0] + rows[1] + rows[2] + rows[3] + epsv);
            rows[0] *= inv_c; rows[1] *= inv_c;
            rows[2] *= inv_c; rows[3] *= inv_c;
        }

        // Write comb output (four aligned 128-bit stores)
        *(device float4*)(comb_out)      = rows[0];
        *(device float4*)(comb_out + 4)  = rows[1];
        *(device float4*)(comb_out + 8)  = rows[2];
        *(device float4*)(comb_out + 12) = rows[3];
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_split_sinkhorn",
        input_names=["mixes", "scale", "base", "eps", "n_rows"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_hc_split_sinkhorn_kernel = _make_hc_split_sinkhorn_kernel()


def hc_split_sinkhorn(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    if _hc_split_sinkhorn_kernel is None or hc_mult != 4:
        return _hc_split_sinkhorn_ops(mixes, scale, base, hc_mult, sinkhorn_iters, eps)

    if not isinstance(eps, mx.array):
        eps = mx.array([eps], dtype=mx.float32)
    n_rows = mixes.size // ((2 + hc_mult) * hc_mult)
    n_rows_arr = mx.array([n_rows], dtype=mx.int32)
    return _hc_split_sinkhorn_kernel(
        inputs=[mixes, scale, base, eps, n_rows_arr],
        template=[("HC", hc_mult), ("ITERS", sinkhorn_iters)],
        grid=((n_rows + 255) & ~255, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult, hc_mult),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


def _make_fused_sparse_attn_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        constexpr int TPH = 4;
        constexpr int DPT = D / TPH;
        uint head = tid / TPH;
        uint lane = tid % TPH;
        uint d_off = lane * DPT;
        if (head >= (uint)H) return;

        int L_v = dims[0], T_v = dims[1], C_v = dims[2], K_v = dims[3];
        uint b = gid / (uint)L_v;
        uint l = gid % (uint)L_v;

        // Load q chunk into registers
        float qr[DPT];
        {
            auto p = q + ((uint64_t)b*H*L_v + head*L_v + l) * D;
            for (int i = 0; i < DPT; i++) qr[i] = float(p[d_off + i]);
        }

        float m_cur = -1e38f, l_sum = 0.f;
        float acc[DPT];
        for (int i = 0; i < DPT; i++) acc[i] = 0.f;
        float sc = float(scale_val[0]);

        // --- Local KV with mask ---
        {
            auto kv_base = local_kv + (uint64_t)b * T_v * D;
            auto m_base = local_mask + ((uint64_t)b * L_v + l) * T_v;
            for (int t = 0; t < T_v; t++) {
                float mv = float(m_base[t]);
                if (mv < -1e9f) continue;

                auto kvp = kv_base + (uint64_t)t * D;
                float kvr[DPT];
                float dot = 0.f;
                for (int i = 0; i < DPT; i++) {
                    kvr[i] = float(kvp[d_off + i]);
                    dot = fma(qr[i], kvr[i], dot);
                }

                float s = dot;
                for (int o = 1; o < TPH; o <<= 1)
                    s += simd_shuffle_xor(s, o);
                s = fma(s, sc, mv);

                float mn = max(m_cur, s);
                float corr = metal::fast::exp(m_cur - mn);
                float p = metal::fast::exp(s - mn);
                l_sum = fma(corr, l_sum, p);
                for (int i = 0; i < DPT; i++)
                    acc[i] = fma(corr, acc[i], p * kvr[i]);
                m_cur = mn;
            }
        }

        // --- Sparse (compressed) KV via index gather ---
        {
            auto ckv = compressed_kv + (uint64_t)b * C_v * D;
            auto ip = topk_idxs + ((uint64_t)b * L_v + l) * K_v;
            for (int k = 0; k < K_v; k++) {
                int idx = int(ip[k]);
                if (idx < 0) continue;

                auto kvp = ckv + (uint64_t)idx * D;
                float kvr[DPT];
                float dot = 0.f;
                for (int i = 0; i < DPT; i++) {
                    kvr[i] = float(kvp[d_off + i]);
                    dot = fma(qr[i], kvr[i], dot);
                }

                float s = dot;
                for (int o = 1; o < TPH; o <<= 1)
                    s += simd_shuffle_xor(s, o);
                s *= sc;

                float mn = max(m_cur, s);
                float corr = metal::fast::exp(m_cur - mn);
                float p = metal::fast::exp(s - mn);
                l_sum = fma(corr, l_sum, p);
                for (int i = 0; i < DPT; i++)
                    acc[i] = fma(corr, acc[i], p * kvr[i]);
                m_cur = mn;
            }
        }

        // --- Attention sink (score = attn_sink[h], value = 0) ---
        {
            float ss = float(attn_sink[head]);
            float mn = max(m_cur, ss);
            float corr = metal::fast::exp(m_cur - mn);
            l_sum = fma(corr, l_sum, metal::fast::exp(ss - mn));
            for (int i = 0; i < DPT; i++) acc[i] *= corr;
            m_cur = mn;
        }

        // --- Normalize and write ---
        {
            float inv_l = 1.f / max(l_sum, 1e-6f);
            auto op = out + ((uint64_t)b*H*L_v + head*L_v + l) * D;
            for (int i = 0; i < DPT; i++)
                store_elem(op[d_off + i], acc[i] * inv_l);
        }
    """

    return mx.fast.metal_kernel(
        name="ds4_fused_sparse_attn",
        input_names=[
            "q", "local_kv", "compressed_kv", "topk_idxs",
            "local_mask", "attn_sink", "scale_val", "dims",
        ],
        output_names=["out"],
        header="template<typename T> inline void store_elem(device T& dst, float v) { dst = T(v); }",
        source=source,
    )


_fused_sparse_attn_kernel = _make_fused_sparse_attn_kernel()


def fused_sparse_attention(
    q: mx.array,
    local_kv: mx.array,
    compressed_kv: mx.array,
    topk_idxs: mx.array,
    local_mask: Optional[mx.array],
    scale: float,
    attn_sink: mx.array,
) -> mx.array:
    """Fused sparse attention: local window + index-gathered compressed KV.

    Uses online softmax (FlashAttention-style) to compute attention over local
    sliding-window KV and per-query sparse compressed KV in a single Metal
    kernel pass.  Avoids materializing the [B, L, topk, D] gathered tensor
    and all intermediate score/exp tensors.
    """
    B, H, L, D = q.shape
    T = local_kv.shape[2]
    C = compressed_kv.shape[1]
    K = topk_idxs.shape[2]

    if _fused_sparse_attn_kernel is None:
        expanded = mx.broadcast_to(
            compressed_kv[:, None, :, :],
            (B, L, C, D),
        )
        idx = topk_idxs[:, :, :, None]
        sparse_kv = mx.take_along_axis(
            expanded, mx.broadcast_to(idx, idx.shape[:-1] + (D,)), axis=2,
        )
        return _split_sparse_attention(
            q, local_kv, sparse_kv, local_mask, scale, attn_sink,
        )

    # Pad mask if local_kv grew beyond mask width
    if local_mask is not None and T > local_mask.shape[-1]:
        pad = mx.zeros(
            local_mask.shape[:-1] + (T - local_mask.shape[-1],),
            dtype=local_mask.dtype,
        )
        local_mask = mx.concatenate([local_mask, pad], axis=-1)

    # Squeeze broadcast dims for contiguous flat access in kernel
    lkv = local_kv.reshape(B, T, D)
    lm = (
        local_mask.reshape(B, L, T).astype(mx.float32)
        if local_mask is not None
        else mx.zeros((B, L, T), dtype=mx.float32)
    )

    dims = mx.array([L, T, C, K], dtype=mx.int32)
    sc = mx.array([scale], dtype=mx.float32)
    sink = attn_sink.astype(mx.float32)

    return _fused_sparse_attn_kernel(
        inputs=[
            q, lkv, compressed_kv,
            topk_idxs.astype(mx.int32),
            lm, sink, sc, dims,
        ],
        template=[("H", H), ("D", D)],
        grid=(B * L * H * 4, 1, 1),
        threadgroup=(H * 4, 1, 1),
        output_shapes=[(B, H, L, D)],
        output_dtypes=[q.dtype],
    )[0]


class HyperConnection(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self._hc_eps = (mx.array([config.hc_eps], dtype=mx.float32),)
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.bfloat16)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def compute_weights(self, x: mx.array):
        B, L, H, D = x.shape
        flat = x.reshape(B, L, H * D)
        flat_f32 = flat.astype(mx.float32)
        rsqrt = mx.rsqrt((flat_f32 * flat_f32).mean(axis=-1, keepdims=True) + self.norm_eps)
        # fn is bfloat16; match flat's dtype so MLX uses a native bf16 GEMV.
        mixes = (flat.astype(self.fn.dtype) @ self.fn.T).astype(mx.float32) * rsqrt
        split_sinkhorn = _hc_split_sinkhorn_ops if self.training else hc_split_sinkhorn
        return split_sinkhorn(
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps if self.training else self._hc_eps[0],
        )

    def collapse(self, x: mx.array):
        pre, post, comb = self.compute_weights(x)
        collapsed = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
        return collapsed.astype(x.dtype), post, comb

    def expand(
        self,
        block_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ):
        y = post[..., None] * block_out[:, :, None, :].astype(mx.float32)
        y = y + mx.matmul(comb.astype(mx.float32), residual.astype(mx.float32))
        return y.astype(block_out.dtype)


class HyperHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.bfloat16
        )
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        B, L, H, D = x.shape
        flat = x.reshape(B, L, H * D)
        flat_f32 = flat.astype(mx.float32)
        rsqrt = mx.rsqrt((flat_f32 * flat_f32).mean(axis=-1, keepdims=True) + self.norm_eps)
        mixes = (flat.astype(self.fn.dtype) @ self.fn.T).astype(mx.float32) * rsqrt
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = layer_idx < config.num_hash_layers
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        flat = x.reshape(-1, self.hidden_dim)
        logits = flat.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        scores = _score_func(logits, self.scoring_func)

        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            inds = self.tid2eid[input_ids.reshape(-1)].astype(mx.int32)
        else:
            biased = scores + self.e_score_correction_bias
            inds = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[
                ..., : self.top_k
            ]

        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.scoring_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        route_shape = (*x.shape[:-1], self.top_k)
        inds = inds.reshape(route_shape)
        weights = weights.reshape(route_shape)
        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype).reshape(x.shape)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class DeepseekV4Cache:
    def __init__(self, sliding_window: int):
        self.local = RotatingKVCache(max_size=sliding_window, keep=0)
        self.compressor_state = {"buffer_x": None, "pooled": None}
        self.indexer_state = {"buffer_x": None, "pooled": None}

    @property
    def offset(self):
        return self.local.offset

    @property
    def keys(self):
        return self.local.keys

    @keys.setter
    def keys(self, value):
        self.local.keys = value

    @property
    def state(self):
        local_state = None if self.local.empty() else self.local.state
        return (
            local_state,
            tuple(self.compressor_state[k] for k in ("buffer_x", "pooled")),
            tuple(self.indexer_state[k] for k in ("buffer_x", "pooled")),
        )

    @state.setter
    def state(self, value):
        local_state, compressor_state, indexer_state = value
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.state = local_state
        self.compressor_state = dict(zip(("buffer_x", "pooled"), compressor_state))
        self.indexer_state = dict(zip(("buffer_x", "pooled"), indexer_state))

    @property
    def meta_state(self):
        return self.local.meta_state

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    def make_mask(self, *args, **kwargs):
        return self.local.make_mask(*args, **kwargs)

    def is_trimmable(self):
        return self.local.is_trimmable()

    def trim(self, n):
        return self.local.trim(n)

    def size(self):
        return self.local.size()

    def empty(self):
        return self.local.empty()

    @property
    def nbytes(self):
        total = self.local.nbytes
        for state in (self.compressor_state, self.indexer_state):
            for value in state.values():
                if value is not None:
                    total += value.nbytes
        return total

    def _branch_state(self, state_key: str):
        return (
            self.indexer_state
            if state_key == "indexer_state"
            else self.compressor_state
        )

    def accumulate_x_windows(
        self,
        x: mx.array,
        state_key: str,
        ratio: int,
        start_pos: int,
    ):
        """Buffer raw hidden states; return the ready portion and pool_base.

        By buffering x instead of (kv, gate), the expensive wkv/wgate GEMVs in
        Compressor are deferred until a full window is ready — saving (ratio-1)/ratio
        of those GEMVs during single-token decode steps.
        """
        state = self._branch_state(state_key)
        buf_x = state["buffer_x"]
        if buf_x is not None and buf_x.shape[1]:
            x = mx.concatenate([buf_x, x], axis=1)
        usable = (x.shape[1] // ratio) * ratio
        state["buffer_x"] = x[:, usable:]
        pool_base = max(0, start_pos) - (buf_x.shape[1] if buf_x is not None else 0)
        return x[:, :usable], pool_base

    def update_pool(self, new_pooled: mx.array, state_key: str) -> mx.array:
        state = self._branch_state(state_key)
        pool = state["pooled"]
        if new_pooled.shape[1] > 0:
            pool = (
                new_pooled
                if pool is None
                else mx.concatenate([pool, new_pooled], axis=1)
            )
            state["pooled"] = pool
        if pool is None:
            pool = mx.zeros(
                (new_pooled.shape[0], 0, new_pooled.shape[-1]), new_pooled.dtype
            )
        return pool


class Compressor(nn.Module):
    # NOTE: Hadamard rotation omitted — assumes unrotated checkpoint.
    # The HF reference applies a fast Walsh-Hadamard transform to KV activations
    # before FP4/FP8 quantization to smooth outlier distributions. This doesn't
    # affect correctness for weights-only quantization, but loading a checkpoint
    # trained with Hadamard-rotated KV states would silently diverge.
    def __init__(self, config: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

    def _overlap_transform(self, x: mx.array, fill_value: float):
        # Build the (B, W, 2R, head_dim) output analytically:
        #   rows [0 : R]  = first-half channels of the PREVIOUS window
        #                   (fill_value for window 0, since there is no prior window)
        #   rows [R : 2R] = second-half channels of the CURRENT window
        # Two concatenates replace the original mx.full + two scatter writes.
        B, W, R, _ = x.shape
        second_half = x[:, :, :, self.head_dim :]                         # (B, W, R, head_dim)
        fill_row    = mx.full((B, 1, R, self.head_dim), fill_value, dtype=x.dtype)
        prev_first  = mx.concatenate(
            [fill_row, x[:, :-1, :, : self.head_dim]], axis=1
        )                                                                   # (B, W, R, head_dim)
        return mx.concatenate([prev_first, second_half], axis=2)           # (B, W, 2R, head_dim)

    def __call__(
        self,
        x: mx.array,
        rope: DeepseekV4RoPE,
        cache: Optional[DeepseekV4Cache],
        start_pos: int,
        state_key: str = "compressor_state",
    ) -> mx.array:
        B, _, _ = x.shape
        if cache is None:
            # Prefill without cache: compute GEMVs for all tokens upfront.
            kv = self.wkv(x)
            gate = self.wgate(x)
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = start_pos
        else:
            # Decode with cache: buffer x and defer wkv/wgate until a full
            # window is ready, saving (ratio-1)/ratio GEMV calls per step.
            ready_x, pool_base = cache.accumulate_x_windows(
                x, state_key, self.compress_ratio, start_pos
            )
            if ready_x.shape[1] == 0:
                return cache.update_pool(
                    mx.zeros((B, 0, self.head_dim), dtype=x.dtype), state_key
                )
            ready_kv = self.wkv(ready_x)
            ready_gate = self.wgate(ready_x)

        if ready_kv.shape[1] == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            W = ready_kv.shape[1] // self.compress_ratio
            kv = ready_kv.reshape(B, W, self.compress_ratio, self.out_dim)
            gate = ready_gate.reshape(
                B, W, self.compress_ratio, self.out_dim
            ) + self.ape.astype(ready_gate.dtype)
            if self.overlap:
                kv = self._overlap_transform(kv, 0.0)
                gate = self._overlap_transform(gate, -float("inf"))
            weights = mx.softmax(gate.astype(mx.float32), axis=2, precise=True).astype(
                kv.dtype
            )
            new_pooled = (kv * weights).sum(axis=2)
            new_pooled = self.norm(new_pooled.astype(x.dtype))
            positions = (
                mx.arange(new_pooled.shape[1], dtype=mx.float32) * self.compress_ratio
                + pool_base
            )
            new_pooled = _apply_partial_rope(
                new_pooled[:, None], rope, positions=positions
            ).squeeze(1)

        if cache is not None:
            return cache.update_pool(new_pooled, state_key)
        return new_pooled


class Indexer(nn.Module):
    # NOTE: Hadamard rotation omitted (see Compressor note above).
    # NOTE: FP4 query simulation omitted. The HF reference quantizes Q to FP4
    # before scoring compressed KV for training/inference parity. Running with
    # full-precision queries gives slightly different top-k selections on edge
    # cases, but the top-k selection is robust enough that this rarely matters.
    def __init__(self, config: ModelArgs, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        rope: DeepseekV4RoPE,
        position_rope: DeepseekV4RoPE,
        cache: Optional[DeepseekV4Cache],
        start_pos: int,
    ):
        B, L, _ = x.shape
        pooled = self.compressor(x, rope, cache, start_pos, state_key="indexer_state")
        if pooled.shape[1] == 0:
            return None

        offset = start_pos
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, position_rope, offset)

        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(
            mx.float32
        )
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        k = min(self.index_topk, pooled.shape[1])
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


def _split_sparse_attention(
    q: mx.array,
    local_kv: mx.array,
    sparse_kv: mx.array,
    local_mask: Optional[mx.array],
    scale: float,
    sinks: Optional[mx.array],
) -> mx.array:
    """Compute attention over local sliding-window KV and per-query sparse KV.

    Avoids materializing [B, H, L, T_local + L*topk] attention scores by
    computing local and sparse attention separately and combining via
    log-sum-exp. Each query position attends only to its own top-k compressed
    keys, not keys selected by other positions.

    Args:
        q: [B, H, L, D] queries.
        local_kv: [B, 1, T_local, D] shared local KV (GQA head dim broadcasts).
        sparse_kv: [B, L, topk, D] per-query-position selected compressed KV.
        local_mask: [B, 1, L, T_local] additive mask for local attention or None.
        scale: attention scale factor.
        sinks: [H] per-head attention sink bias or None. Adds a virtual token
            with this score and zero value to the softmax denominator.
    """
    q_f = q.astype(mx.float32)

    # Local scores: [B, H, L, T_local]
    local_scores = (q_f @ local_kv.astype(mx.float32).swapaxes(-1, -2)) * scale
    if local_mask is not None:
        local_scores = local_scores + local_mask

    # Sparse scores: [B, H, L, topk]
    # [B, H, L, 1, D] @ [B, 1, L, D, topk] -> [B, H, L, 1, topk] -> squeeze
    sparse_kv_f = sparse_kv.astype(mx.float32)
    sparse_scores = (
        q_f[:, :, :, None, :] @ sparse_kv_f[:, None].swapaxes(-1, -2)
    ).squeeze(3) * scale

    # Combined softmax via log-sum-exp
    local_max = local_scores.max(axis=-1, keepdims=True)
    sparse_max = sparse_scores.max(axis=-1, keepdims=True)
    m = mx.maximum(local_max, sparse_max)

    # Attention sinks: virtual token with score=sink[h], value=0
    if sinks is not None:
        sink_scores = sinks[None, :, None, None].astype(mx.float32)
        m = mx.maximum(m, sink_scores)

    local_exp = mx.exp(local_scores - m)
    sparse_exp = mx.exp(sparse_scores - m)
    denom = local_exp.sum(axis=-1, keepdims=True) + sparse_exp.sum(axis=-1, keepdims=True)
    if sinks is not None:
        denom = denom + mx.exp(sink_scores - m)

    inv_denom = mx.reciprocal(denom)

    # Local values: [B, H, L, T_local] @ [B, 1, T_local, D] -> [B, H, L, D]
    out = (local_exp * inv_denom) @ local_kv.astype(mx.float32)
    # Sparse values: [B, H, L, 1, topk] @ [B, 1, L, topk, D] -> [B, H, L, 1, D]
    out = out + (
        (sparse_exp * inv_denom)[:, :, :, None, :] @ sparse_kv_f[:, None]
    ).squeeze(3)

    return out.astype(q.dtype)


class V4Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        rope_theta = (
            config.compress_rope_theta if self.compress_ratio else config.rope_theta
        )
        rope_scaling = config.rope_scaling if self.compress_ratio else None
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            rope_theta,
            rope_scaling,
            config.max_position_embeddings,
        )
        self.compress_rope = self.rope
        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(config, self.compress_ratio)

    def _grouped_output_projection(self, out: mx.array) -> mx.array:
        B, L = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.o_groups
        out = out.reshape(B, L, self.o_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            out = out.transpose(2, 0, 1, 3)
            weight = self.wo_a.weight.reshape(self.o_groups, self.o_lora_rank, -1)[
                :, None
            ]
            scales = self.wo_a.scales.reshape(self.o_groups, self.o_lora_rank, -1)[
                :, None
            ]
            biases = (
                None
                if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.o_groups, self.o_lora_rank, -1)[
                    :, None
                ]
            )
            out = mx.quantized_matmul(
                out,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=self.wo_a.group_size,
                bits=self.wo_a.bits,
                mode=self.wo_a.mode,
            )
            out = out.transpose(1, 2, 0, 3).reshape(
                B, L, self.o_groups * self.o_lora_rank
            )
            if "bias" in self.wo_a:
                out = out + self.wo_a.bias
            return out

        weight = self.wo_a.weight.reshape(self.o_groups, self.o_lora_rank, group_feat)
        out = mx.einsum("bsgd,grd->bsgr", out, weight)
        out = out.reshape(B, L, self.o_groups * self.o_lora_rank)
        if "bias" in self.wo_a:
            out = out + self.wo_a.bias
        return out

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache
        if isinstance(cache, DeepseekV4Cache):
            local_cache = cache

        offset = local_cache.offset if local_cache is not None else 0
        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        if _q_norm_kernel is not None:
            q = _q_norm_kernel(
                inputs=[
                    q,
                    mx.array([self.config.rms_norm_eps], dtype=mx.float32),
                    mx.array([self.head_dim, L, self.n_heads], dtype=mx.int32),
                ],
                grid=(B * L * self.n_heads * 32, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[q.shape],
                output_dtypes=[q.dtype],
            )[0]
        else:
            q = (
                q * mx.rsqrt(
                    (q.astype(mx.float32) ** 2).mean(axis=-1, keepdims=True)
                    + self.config.rms_norm_eps
                )
            ).astype(x.dtype)
        q = q.transpose(0, 2, 1, 3)
        kv = self.kv_norm(self.wkv(x)).reshape(B, L, 1, self.head_dim)
        kv = kv.transpose(0, 2, 1, 3)

        q = _apply_partial_rope(q, self.rope, offset)
        kv = _apply_partial_rope(kv, self.rope, offset)

        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        full_kv = kv

        fused_out = None
        if self.compress_ratio:
            v4_cache = cache if isinstance(cache, DeepseekV4Cache) else None
            pooled = self.compressor(x, self.compress_rope, v4_cache, offset)
            if pooled.shape[1] > 0:
                if hasattr(self, "indexer") and L > 1:
                    # Prefill only: run the indexer to get sparse top-k indices.
                    # For decode (L==1), C <= index_topk always so the indexer
                    # would select all tokens anyway — skip it and use pooled[:, None].
                    topk = self.indexer(
                        x, q_residual, self.compress_rope, self.rope, v4_cache, offset
                    )
                    if topk is not None:
                        fused_out = fused_sparse_attention(
                            q, full_kv, pooled, topk, mask,
                            self.scale, self.attn_sink.astype(q.dtype),
                        )
                    else:
                        full_kv = mx.concatenate([full_kv, pooled[:, None]], axis=2)
                else:
                    full_kv = mx.concatenate([full_kv, pooled[:, None]], axis=2)

        if fused_out is not None:
            out = fused_out
        else:
            if mask is not None and full_kv.shape[2] > mask.shape[-1]:
                pad = mx.zeros(
                    mask.shape[:-1] + (full_kv.shape[2] - mask.shape[-1],),
                    dtype=mask.dtype,
                )
                mask = mx.concatenate([mask, pad], axis=-1)
            out = scaled_dot_product_attention(
                q,
                full_kv,
                full_kv,
                cache=local_cache,
                scale=self.scale,
                mask=mask,
                sinks=self.attn_sink.astype(q.dtype),
            )
        out = _apply_partial_rope(out, self.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = V4Attention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc.collapse(h)
        x = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        h = self.attn_hc.expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc.collapse(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return self.ffn_hc.expand(x, residual, post, comb)


class DeepseekV4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        # generate_step adds [None] to the prompt, which can produce (1, 1, L)
        # when the caller already passes (1, L). Flatten to (B, L).
        if inputs.ndim != 2:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache.local
            if isinstance(first_cache, DeepseekV4Cache)
            else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, mask, layer_cache, inputs)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, DeepseekV4Cache):
                cache_item = cache_item.local
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(self.hc_head(h))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            # attn_sink and correction bias must stay float32 (small, precision-sensitive).
            # HC base/scale are tiny f32 bias/gain arrays used inside sinkhorn — keep f32.
            # HC fn weights are large (mix × hidden) projection matrices: cast to bfloat16
            # to halve their memory footprint; compute_weights re-casts if needed.
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or k.endswith(
                    (
                        ".attn_hc.base",
                        ".attn_hc.scale",
                        ".ffn_hc.base",
                        ".ffn_hc.scale",
                        ".hc_head.base",
                        ".hc_head.scale",
                    )
                )
            )

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith(
                (
                    ".ffn.switch_mlp.gate_proj",
                    ".ffn.switch_mlp.up_proj",
                    ".ffn.switch_mlp.down_proj",
                )
            ):
                return _default_quantization()
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attn.compress_ratio:
                caches.append(DeepseekV4Cache(self.args.sliding_window))
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers

        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        def scale_to_float(scale: mx.array) -> mx.array:
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        def dequant_fp8(weight: mx.array, scale: mx.array, block_size: int = 128):
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            scale = scale_to_float(scale)
            m, n = weight.shape
            pad_m = (-m) % block_size
            pad_n = (-n) % block_size
            weight = mx.pad(weight, ((0, pad_m), (0, pad_n)))
            weight = weight.reshape(
                (m + pad_m) // block_size,
                block_size,
                (n + pad_n) // block_size,
                block_size,
            )
            weight = (weight * scale[:, None, :, None]).reshape(m + pad_m, n + pad_n)
            return weight[:m, :n].astype(mx.bfloat16)

        def pack_fp4(weight: mx.array):
            packed = weight.astype(mx.uint8)
            *dims, packed_in = packed.shape
            packed = packed.reshape(*dims, packed_in // 4, 4).astype(mx.uint32)
            shifts = mx.array([0, 8, 16, 24], dtype=mx.uint32)
            return ((packed << shifts).sum(axis=-1)).astype(mx.uint32)

        new_weights = {}
        for k, v in weights.items():
            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[wk] = pack_fp4(weight)
                new_weights[wk[: -len(".weight")] + ".scales"] = v.astype(mx.uint8)
            elif weight.dtype == mx.uint8:
                new_weights[wk] = dequant_fp8(weight, v)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "embed.scales": "model.embed_tokens.scales",
            "embed.biases": "model.embed_tokens.biases",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "head.scales": "lm_head.scales",
            "head.biases": "lm_head.biases",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales", "biases"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    pre_stacked_key = f"{prefix}.{src}.{suffix}"
                    dst_key = (
                        f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
                    )
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[dst_key] = mx.stack(stacked)
                    elif pre_stacked_key in weights:
                        weights[dst_key] = weights.pop(pre_stacked_key)

        # Stack grouped wo_a.0..N into single wo_a (concat along output dim)
        o_groups = self.args.o_groups
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.attn.wo_a"
            for suffix in ("weight", "scales", "biases"):
                key0 = f"{prefix}.0.{suffix}"
                if key0 in weights:
                    parts = [
                        weights.pop(f"{prefix}.{g}.{suffix}")
                        for g in range(o_groups)
                    ]
                    weights[f"{prefix}.{suffix}"] = mx.concatenate(parts, axis=0)

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            layer.attn.wq_b = shard_linear(
                layer.attn.wq_b, "all-to-sharded", group=group
            )
            layer.attn.wo_b = shard_linear(
                layer.attn.wo_b, "sharded-to-all", group=group
            )
            layer.attn.n_heads //= N

            layer.ffn.sharding_group = group
            shard_inplace(
                layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
            )
            shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(layer.ffn.switch_mlp.up_proj, "all-to-sharded", group=group)
