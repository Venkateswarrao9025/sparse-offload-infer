"""Thin typed wrappers over the soinfer CUDA extension (soinfer._C)."""
import torch

from . import _C


def add_one(x: torch.Tensor) -> torch.Tensor:
    """out[i] = x[i] + 1. float32 CUDA tensors only. M0 toolchain smoke test."""
    return _C.add_one(x)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """out = a + b, elementwise. float32 CUDA tensors only. M1."""
    return _C.vector_add(a, b)


def strided_copy(x: torch.Tensor, stride: int) -> torch.Tensor:
    """out[i] = x[i*stride] for i in [0, len(x)//stride). M1 coalescing sweep."""
    return _C.strided_copy(x, stride)


def reduce_naive_atomic(x: torch.Tensor) -> torch.Tensor:
    """Sum of x via v1: every thread atomicAdd's its element to the output. M1."""
    return _C.reduce_naive_atomic(x)


def reduce_shared_tree(x: torch.Tensor) -> torch.Tensor:
    """Sum of x via v2: per-block shared-memory tree reduction. M1."""
    return _C.reduce_shared_tree(x)


def reduce_warp_shuffle(x: torch.Tensor) -> torch.Tensor:
    """Sum of x via v3: per-block warp-shuffle reduction. M1."""
    return _C.reduce_warp_shuffle(x)


def reduce_vectorized(x: torch.Tensor) -> torch.Tensor:
    """Sum of x via v4: vectorized float4 loads + warp-shuffle. numel must be a multiple of 4. M1."""
    return _C.reduce_vectorized(x)


def transpose_naive(x: torch.Tensor) -> torch.Tensor:
    """Square matrix transpose v1: naive, one thread per element. M1."""
    return _C.transpose_naive(x)


def transpose_unpadded(x: torch.Tensor) -> torch.Tensor:
    """Square matrix transpose v2: shared-memory tiled, bank-conflicted. M1."""
    return _C.transpose_unpadded(x)


def transpose_padded(x: torch.Tensor) -> torch.Tensor:
    """Square matrix transpose v3: shared-memory tiled, padded to avoid bank conflicts. M1."""
    return _C.transpose_padded(x)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim. x: [rows, hidden] half, weight: [hidden] half. hidden must be even. M2."""
    return _C.rmsnorm(x, weight, eps)


def softmax_twopass(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax v1: naive two-pass statistics (max pass, then sum pass). x: [rows, cols] half. M2."""
    return _C.softmax_twopass(x)


def softmax_online(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax v2: online single-pass statistics (FlashAttention rescaling recurrence). M2."""
    return _C.softmax_online(x)


def gemv_fp16_v1(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = W @ x. v1: one thread per output row. W: [N,K] half, x: [K] half. M2."""
    return _C.gemv_fp16_v1(W, x)


def gemv_fp16_v2(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = W @ x. v2: one warp per row, warp-shuffle reduce. M2."""
    return _C.gemv_fp16_v2(W, x)


def gemv_fp16_v3(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = W @ x. v3: v2 + float4-vectorized loads. Requires K % 8 == 0. M2."""
    return _C.gemv_fp16_v3(W, x)


def gemv_fp16_v4_splitk(W: torch.Tensor, x: torch.Tensor, split: int) -> torch.Tensor:
    """y = W @ x. v4: split-K with atomics -- helps when N is too small to saturate the GPU. M2."""
    return _C.gemv_fp16_v4_splitk(W, x, split)


def gemv_w8a16(Wq: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, group_size: int) -> torch.Tensor:
    """y = dequant(Wq, scale) @ x. Wq: [N,K] int8 (symmetric, see soinfer.quant.formats). scale: [N,num_groups]
    or [1,num_groups] fp32 (broadcast for per_tensor). group_size: K for per_tensor/per_channel, else the
    quant group size (scale[row, k // group_size] is used for element k). M4."""
    return _C.gemv_w8a16(Wq, scale, x, group_size)


def gemv_w4a16_group(Wq_packed: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, K: int, group_size: int) -> torch.Tensor:
    """y = dequant(Wq_packed, scale) @ x. Wq_packed: [N, ceil(K/8)*4] uint8, AWQ-order-packed INT4 (see
    soinfer.quant.pack.pack_int4 / csrc/include/layout.h). scale: [N, num_groups] fp32, group_size a positive
    multiple of 8. Scalar dequant, register-cached group scale. M4."""
    return _C.gemv_w4a16_group(Wq_packed, scale, x, K, group_size)


def gemv_w4a16_group_lop3(Wq_packed: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, K: int, group_size: int) -> torch.Tensor:
    """Same contract as gemv_w4a16_group, but dequantizes via FP16 bit-pattern construction instead of
    int->float conversion instructions (PROJECT_SPEC.md M4 task 3). M4."""
    return _C.gemv_w4a16_group_lop3(Wq_packed, scale, x, K, group_size)


def swiglu_gate_up(gate_W: torch.Tensor, up_W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """h = silu(gate_W @ x) * (up_W @ x). gate_W, up_W: [I,H] half, same shape. x: [H] half. H must be a
    multiple of 8. Fuses the gate/up GEMVs (one shared pass over x) and the SiLU+multiply into one kernel --
    only h round-trips to global memory, not separate gate/up pre-activation buffers. M5."""
    return _C.swiglu_gate_up(gate_W, up_W, x)


def fused_swiglu_mlp(gate_W: torch.Tensor, up_W: torch.Tensor, down_W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = down_W @ (silu(gate_W @ x) * (up_W @ x)) -- full SwiGLU MLP. gate_W, up_W: [I,H] half, down_W:
    [H,I] half, x: [H] half. Uses swiglu_gate_up for the fused gate/up/SiLU pass, then gemv_fp16_v3 for the
    down projection (down needs the complete intermediate vector, so it isn't fused into the same kernel --
    see csrc/kernels/swiglu_fused.cuh). M5."""
    h = swiglu_gate_up(gate_W, up_W, x)
    return gemv_fp16_v3(down_W, h)


def concat_qkv_weights(Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor) -> torch.Tensor:
    """Build the [q_dim + 2*kv_dim, H] concatenated weight fused_qkv_projection expects. Call once at
    model-load time, not per token -- the whole point is to amortize the concat cost across every
    decode step. M5."""
    return torch.cat([Wq, Wk, Wv], dim=0).contiguous()


def fused_qkv_projection(
    qkv_W: torch.Tensor, x: torch.Tensor, q_dim: int, kv_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k, v = split(qkv_W @ x). qkv_W: [q_dim + 2*kv_dim, H] half, built once via
    concat_qkv_weights. Unlike swiglu_gate_up (where gate and up share a downstream elementwise
    combine, so fusing the actual dot-product compute is the win), Q/K/V don't share any compute
    -- the win here is purely launch-overhead and traffic-planning: one GEMV kernel launch and one
    pass over x instead of three, which is what "one kernel, three outputs, one pass over x" (M5
    task 2) means in the batch-1 decode regime where launches are the bottleneck, not FLOPs.
    Reuses gemv_fp16_v3 rather than a new kernel -- no new compute pattern is needed once the
    weights are concatenated. M5."""
    qkv = gemv_fp16_v3(qkv_W, x)
    q, k, v = qkv.split([q_dim, kv_dim, kv_dim])
    return q, k, v


def topk_threshold_select(abs_g: torch.Tensor, k: int) -> torch.Tensor:
    """Returns the k indices of abs_g's largest values (unordered). abs_g: 1D float32 CUDA
    tensor, non-negative (e.g. |gate_proj(x)|, the SwiGLU gate activation magnitude M7 selects
    channels from). Single kernel launch: binary-searches a threshold within one block (no
    host round-trips between search iterations -- see csrc/kernels/topk_select.cuh), then
    compacts qualifying indices. M7."""
    indices, _count = _C.topk_threshold_select(abs_g, k)
    return indices


def gather_rows_staged(matrix: torch.Tensor, indices: torch.Tensor, staging: torch.Tensor,
                        gpu_dst: torch.Tensor) -> None:
    """M7 task 2, coalesced variant: memcpy each selected row of `matrix` (pinned CPU uint8,
    [num_rows, row_nbytes], e.g. PinnedWeightStore.matrix_view(handle)) into `staging` (pinned
    CPU uint8, >= k*row_nbytes), then a single H2D cudaMemcpyAsync of the whole staged block into
    `gpu_dst` (CUDA uint8, >= k*row_nbytes). Async on the current stream -- caller must
    synchronize before reading gpu_dst. indices: 1D int64 CPU tensor."""
    _C.gather_rows_staged(matrix, indices, staging, gpu_dst)


def gather_rows_naive(matrix: torch.Tensor, indices: torch.Tensor, gpu_dst: torch.Tensor) -> None:
    """M7 task 2, naive baseline: one cudaMemcpyAsync per selected row, straight from its
    (scattered) offset in `matrix` (pinned CPU uint8) to its slot in `gpu_dst` (CUDA uint8).
    Same arguments and async-on-current-stream semantics as gather_rows_staged, minus the
    staging buffer -- exists to benchmark against it (see PROJECT_SPEC.md M7 task 2)."""
    _C.gather_rows_naive(matrix, indices, gpu_dst)


def gemv_w4a16_sparse_accumulate(Wq_selected: torch.Tensor, scale_selected: torch.Tensor,
                                  h_selected: torch.Tensor, H: int, group_size: int) -> torch.Tensor:
    """M7 task 3, 'down' direction: y[H] = sum_i h_selected[i] * W_T[i, :], over the k selected
    (already row-gathered via gather_rows_*) rows of down_proj stored TRANSPOSED
    ([intermediate_size, hidden_size] instead of nn.Linear's usual [hidden_size,
    intermediate_size]) so channel selection is a row gather here too, same as up_proj. The
    'up' direction needs no new kernel: it's gemv_w4a16_group_lop3 applied directly to
    up_proj's k gathered rows (N=k instead of N=intermediate_size), since up_proj is already
    row-indexed by intermediate channel."""
    return _C.gemv_w4a16_sparse_accumulate(Wq_selected, scale_selected, h_selected, H, group_size)


def precompute_rope_cos_sin(head_dim: int, theta: float, pos: int, device, dtype=torch.float32):
    """cos/sin for RoPE at absolute position `pos`, matching HF's Qwen3RotaryEmbedding exactly:
    inv_freq[i] = 1/theta^(2i/head_dim) for i in [0, head_dim/2), angle_i = pos*inv_freq[i]. M5."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    angles = pos * inv_freq
    return angles.cos().to(dtype).contiguous(), angles.sin().to(dtype).contiguous()


def apply_rope(x: torch.Tensor, cos_vals: torch.Tensor, sin_vals: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to x [num_heads, head_dim] half IN PLACE (rotate-half convention, matches HF's
    apply_rotary_pos_emb exactly -- see csrc/kernels/rope.cuh's derivation). cos_vals/sin_vals:
    [head_dim/2] float32, from precompute_rope_cos_sin. Returns x for chaining. M5."""
    _C.rope_apply(x, cos_vals, sin_vals)
    return x


def qk_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3-style per-head QK-norm: plain RMSNorm over head_dim, independently per head. x:
    [num_heads, head_dim] half, weight: [head_dim] half. Reuses the M2 rmsnorm kernel directly --
    Qwen3Attention's q_norm/k_norm IS exactly RMSNorm(head_dim) applied per head, no new kernel
    needed. M5."""
    return rmsnorm(x, weight, eps)


def kv_cache_append(k_cache: torch.Tensor, v_cache: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, pos: int) -> None:
    """Writes k_new/v_new (each [num_kv_heads, head_dim] half) into k_cache/v_cache (each
    [num_kv_heads, max_seq_len, head_dim] half) at sequence position `pos`, in place. Contiguous
    layout (paged/block-table layout is a stretch goal, not implemented). M5."""
    _C.kv_cache_append(k_cache, v_cache, k_new, v_new, pos)


def decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cur_len: int) -> torch.Tensor:
    """out = attention(q, k_cache[:, :cur_len], v_cache[:, :cur_len]) for a single query token.
    q: [num_q_heads, head_dim] half, already RoPE-rotated (and QK-normed, for architectures like
    Qwen3 that use it) by the caller -- this kernel is the attention math only (scores, online
    softmax, weighted V), not positional encoding. k_cache/v_cache: [num_kv_heads, max_seq_len,
    head_dim] half; num_q_heads must be a multiple of num_kv_heads (GQA), grouped the same way as
    HF's `repeat_kv` (query head qh reads KV head qh // (num_q_heads // num_kv_heads)). M5."""
    return _C.decode_attention(q, k_cache, v_cache, cur_len)
