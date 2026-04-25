"""
Triton Kernel for Gated Delta Net Decoding.

Inputs:
  q:       bfloat16 [batch_size, seq_len, num_q_heads, head_size]
  k:       bfloat16 [batch_size, seq_len, num_k_heads, head_size]
  v:       bfloat16 [batch_size, seq_len, num_v_heads, head_size]
  state:   float32  [batch_size, num_v_heads, head_size, head_size]  (V, K layout)
  A_log:   float32  [num_v_heads]
  a:       bfloat16 [batch_size, seq_len, num_v_heads]
  dt_bias: float32  [num_v_heads]
  b:       bfloat16 [batch_size, seq_len, num_v_heads]
  scale:   float32  scalar

Outputs:
  output:    bfloat16 [batch_size, seq_len, num_v_heads, head_size]
  new_state: float32  [batch_size, num_v_heads, head_size, head_size]

Note: GVA configuration where num_v_heads > num_q_heads = num_k_heads
      q and k are repeated to match num_v_heads
      State is in V,K layout (last dimension is K)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_V=128 (1 V-tile): full V at once, vary K tiling
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=1),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=2),
        # BLOCK_V=64 (2 V-tiles): good balance of parallelism and register use
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 32},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_V': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=2),
        # BLOCK_V=32 (4 V-tiles): max parallelism, smallest register footprint
        triton.Config({'BLOCK_V': 32,  'BLOCK_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_V': 32,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 32,  'BLOCK_K': 64},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_V': 32,  'BLOCK_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_V': 32,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 32,  'BLOCK_K': 32},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_V': 32,  'BLOCK_K': 32},  num_warps=2, num_stages=2),
        # BLOCK_V=16 (8 V-tiles): highest parallelism, lowest register per program
        triton.Config({'BLOCK_V': 16,  'BLOCK_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_V': 16,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 16,  'BLOCK_K': 64},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_V': 16,  'BLOCK_K': 32},  num_warps=1, num_stages=1),
        triton.Config({'BLOCK_V': 16,  'BLOCK_K': 32},  num_warps=2, num_stages=1),
    ],
    key=['HEAD_SIZE', 'V_DIM'],
)
@triton.jit
def gdn_decode_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr,
    STATE_ptr,
    A_log_ptr, A_ptr, DT_bias_ptr, B_ptr,
    # Output pointers
    OUT_ptr, NEW_STATE_ptr,
    # Strides for q: [batch_size, seq_len, num_q_heads, head_size]
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    # Strides for k: [batch_size, seq_len, num_k_heads, head_size]
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    # Strides for v: [batch_size, seq_len, num_v_heads, head_size]
    stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    # Strides for state: [batch_size, num_v_heads, V, K] - note V,K layout
    stride_state_b, stride_state_h, stride_state_v, stride_state_k,
    # Strides for a: [batch_size, seq_len, num_v_heads]
    stride_a_b, stride_a_s, stride_a_h,
    # Strides for b: [batch_size, seq_len, num_v_heads]
    stride_b_b, stride_b_s, stride_b_h,
    # Strides for output: [batch_size, seq_len, num_v_heads, head_size]
    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
    # Strides for new_state: [batch_size, num_v_heads, V, K]
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    # Scale
    scale,
    # Head ratio for GVA (num_v_heads // num_q_heads)
    head_ratio,
    V_DIM,
    # Constexpr dimensions
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Gated Delta Net decoding kernel with BLOCK_V x BLOCK_K tiling.

    Each program handles one (batch, v_head, v_tile, seq) combination.
    The K dimension is processed in BLOCK_K chunks via a loop to reduce
    register pressure and improve occupancy.
    """
    pid_batch = tl.program_id(0)
    pid_v_head = tl.program_id(1)
    pid_compound = tl.program_id(2)

    num_v_tiles = tl.cdiv(V_DIM, BLOCK_V)
    pid_v_tile = pid_compound % num_v_tiles
    pid_seq = pid_compound // num_v_tiles

    pid_qk_head = pid_v_head // head_ratio

    # --- gate scalars ---
    # Mixed precision gate path: keep recurrent state math in fp32,
    # but reduce gate input precision to lower instruction/memory pressure.
    A_log_val_bf16 = tl.load(A_log_ptr + pid_v_head).to(tl.bfloat16)
    dt_bias_val_bf16 = tl.load(DT_bias_ptr + pid_v_head).to(tl.bfloat16)
    a_val_bf16 = tl.load(
        A_ptr + pid_batch * stride_a_b + pid_seq * stride_a_s + pid_v_head * stride_a_h
    ).to(tl.bfloat16)
    b_val_bf16 = tl.load(
        B_ptr + pid_batch * stride_b_b + pid_seq * stride_b_s + pid_v_head * stride_b_h
    ).to(tl.bfloat16)

    x_bf16 = a_val_bf16 + dt_bias_val_bf16
    x = x_bf16.to(tl.float32)
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    alpha = tl.exp(-tl.exp(A_log_val_bf16.to(tl.float32)) * softplus_x)
    beta = tl.sigmoid(b_val_bf16.to(tl.float32))

    # --- V-dimension tile offsets ---
    v_off = pid_v_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_off < V_DIM

    # Load v vector tile
    v_ptrs = V_ptr + pid_batch * stride_v_b + pid_seq * stride_v_s + pid_v_head * stride_v_h + v_off * stride_v_d
    v_vec = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)  # [BLOCK_V]

    # Base pointers for q, k (shared across K chunks)
    q_base = Q_ptr + pid_batch * stride_q_b + pid_seq * stride_q_s + pid_qk_head * stride_q_h
    k_base = K_ptr + pid_batch * stride_k_b + pid_seq * stride_k_s + pid_qk_head * stride_k_h

    # Base pointers for state load / new_state store
    state_base = (STATE_ptr + pid_batch * stride_state_b + pid_v_head * stride_state_h)
    ns_base = (NEW_STATE_ptr + pid_batch * stride_ns_b + pid_v_head * stride_ns_h)

    # Accumulators across K chunks
    old_v_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    old_out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    qk_dot = tl.zeros((), dtype=tl.float32)

    if BLOCK_K == HEAD_SIZE:
        # --- Single-pass: full K in one block, no loop ---
        k_off = tl.arange(0, BLOCK_K)
        k_mask = k_off < HEAD_SIZE

        q_chunk = tl.load(q_base + k_off * stride_q_d, mask=k_mask, other=0.0).to(tl.float32)
        k_chunk = tl.load(k_base + k_off * stride_k_d, mask=k_mask, other=0.0).to(tl.float32)
        state_ptrs = (state_base
                     + v_off[:, None] * stride_state_v
                     + k_off[None, :] * stride_state_k)
        state_chunk = tl.load(state_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        old_state_chunk = alpha * state_chunk
        old_v_acc = tl.sum(old_state_chunk * k_chunk[None, :], axis=1)
        old_out_acc = tl.sum(old_state_chunk * q_chunk[None, :], axis=1)
        qk_dot = tl.sum(q_chunk * k_chunk, axis=0)
        new_v = beta * v_vec + (1.0 - beta) * old_v_acc
        delta_v = new_v - old_v_acc
        new_state_chunk = old_state_chunk + k_chunk[None, :] * delta_v[:, None]
        out_acc = old_out_acc + qk_dot * delta_v

        ns_ptrs = (ns_base
                  + v_off[:, None] * stride_ns_v
                  + k_off[None, :] * stride_ns_k)
        tl.store(ns_ptrs, new_state_chunk.to(tl.float32), mask=v_mask[:, None] & k_mask[None, :])
    else:
        # --- Two-pass: loop over K dimension in BLOCK_K chunks ---
        for k_start in tl.static_range(0, HEAD_SIZE, BLOCK_K):
            k_off = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_off < HEAD_SIZE

            q_chunk = tl.load(q_base + k_off * stride_q_d, mask=k_mask, other=0.0).to(tl.float32)
            k_chunk = tl.load(k_base + k_off * stride_k_d, mask=k_mask, other=0.0).to(tl.float32)
            state_ptrs = (state_base
                         + v_off[:, None] * stride_state_v
                         + k_off[None, :] * stride_state_k)
            state_chunk = tl.load(state_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

            old_state_chunk = alpha * state_chunk
            old_v_acc += tl.sum(old_state_chunk * k_chunk[None, :], axis=1)
            old_out_acc += tl.sum(old_state_chunk * q_chunk[None, :], axis=1)
            qk_dot += tl.sum(q_chunk * k_chunk, axis=0)

        new_v = beta * v_vec + (1.0 - beta) * old_v_acc
        delta_v = new_v - old_v_acc
        out_acc = old_out_acc + qk_dot * delta_v

        for k_start in tl.static_range(0, HEAD_SIZE, BLOCK_K):
            k_off = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_off < HEAD_SIZE

            k_chunk = tl.load(k_base + k_off * stride_k_d, mask=k_mask, other=0.0).to(tl.float32)
            state_ptrs = (state_base
                         + v_off[:, None] * stride_state_v
                         + k_off[None, :] * stride_state_k)
            state_chunk = tl.load(state_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

            old_state_chunk = alpha * state_chunk
            new_state_chunk = old_state_chunk + k_chunk[None, :] * delta_v[:, None]

            ns_ptrs = (ns_base
                      + v_off[:, None] * stride_ns_v
                      + k_off[None, :] * stride_ns_k)
            tl.store(ns_ptrs, new_state_chunk.to(tl.float32), mask=v_mask[:, None] & k_mask[None, :])

    # Store output tile
    out_tile = scale * out_acc
    out_ptrs = OUT_ptr + pid_batch * stride_o_b + pid_seq * stride_o_s + pid_v_head * stride_o_h + v_off * stride_o_d
    tl.store(out_ptrs, out_tile.to(tl.bfloat16), mask=v_mask)


def gdn_decode(
    q: torch.Tensor,        # [batch_size, seq_len, num_q_heads, head_size]
    k: torch.Tensor,        # [batch_size, seq_len, num_k_heads, head_size]
    v: torch.Tensor,        # [batch_size, seq_len, num_v_heads, head_size]
    state: torch.Tensor,    # [batch_size, num_v_heads, head_size, head_size]
    A_log: torch.Tensor,    # [num_v_heads]
    a: torch.Tensor,        # [batch_size, seq_len, num_v_heads]
    dt_bias: torch.Tensor,  # [num_v_heads]
    b: torch.Tensor,        # [batch_size, seq_len, num_v_heads]
    scale: float,           # scalar
    output: torch.Tensor = None,
    new_state: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[2]
    head_ratio = num_v_heads // num_q_heads

    if state is None:
        state = torch.zeros(batch_size, num_v_heads, head_size, head_size,
                           dtype=torch.float32, device=q.device)
    if scale is None or scale == 0.0:
        scale = 1.0 / (head_size ** 0.5)
    if output is None:
        output = torch.empty(batch_size, seq_len, num_v_heads, head_size,
                            dtype=torch.bfloat16, device=q.device)
    if new_state is None:
        new_state = torch.empty(batch_size, num_v_heads, head_size, head_size,
                                dtype=torch.float32, device=q.device)

    def grid(META):
        num_v_tiles = triton.cdiv(head_size, META['BLOCK_V'])
        return (batch_size, num_v_heads, num_v_tiles * seq_len)

    gdn_decode_kernel[grid](
        q, k, v,
        state,
        A_log, a, dt_bias, b,
        output, new_state,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        scale,
        head_ratio,
        head_size,
        HEAD_SIZE=head_size,
    )

    return output, new_state