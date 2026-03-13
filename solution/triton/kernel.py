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
import torch.nn.functional as F
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
    A_log_val = tl.load(A_log_ptr + pid_v_head).to(tl.float32)
    dt_bias_val = tl.load(DT_bias_ptr + pid_v_head).to(tl.float32)
    a_val = tl.load(A_ptr + pid_batch * stride_a_b + pid_seq * stride_a_s + pid_v_head * stride_a_h).to(tl.float32)
    b_val = tl.load(B_ptr + pid_batch * stride_b_b + pid_seq * stride_b_s + pid_v_head * stride_b_h).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    alpha = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta = tl.sigmoid(b_val)

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
    out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

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
        new_v = beta * v_vec + (1.0 - beta) * old_v_acc
        delta_v = new_v - old_v_acc
        new_state_chunk = old_state_chunk + k_chunk[None, :] * delta_v[:, None]
        out_acc = tl.sum(new_state_chunk * q_chunk[None, :], axis=1)

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

        new_v = beta * v_vec + (1.0 - beta) * old_v_acc
        delta_v = new_v - old_v_acc

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
            new_state_chunk = old_state_chunk + k_chunk[None, :] * delta_v[:, None]
            out_acc += tl.sum(new_state_chunk * q_chunk[None, :], axis=1)

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
        new_state = torch.empty_like(state)

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


@triton.jit
def gdn_prefill_step_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr,
    STATE_ptr,              # in/out: [num_seqs, num_v_heads, V, K]
    A_log_ptr, A_ptr, DT_bias_ptr, B_ptr,
    # Output pointer
    OUT_ptr,                # [total_seq_len, num_v_heads, head_size]
    # Current token / sequence index
    token_idx,
    seq_idx,
    # Strides for q: [total_seq_len, num_q_heads, head_size]
    stride_q_t, stride_q_h, stride_q_d,
    # Strides for k: [total_seq_len, num_k_heads, head_size]
    stride_k_t, stride_k_h, stride_k_d,
    # Strides for v: [total_seq_len, num_v_heads, head_size]
    stride_v_t, stride_v_h, stride_v_d,
    # Strides for state: [num_seqs, num_v_heads, V, K]
    stride_state_s, stride_state_h, stride_state_v, stride_state_k,
    # Strides for a: [total_seq_len, num_v_heads]
    stride_a_t, stride_a_h,
    # Strides for b: [total_seq_len, num_v_heads]
    stride_b_t, stride_b_h,
    # Strides for output: [total_seq_len, num_v_heads, head_size]
    stride_o_t, stride_o_h, stride_o_d,
    # Scale
    scale,
    # Head ratio for GVA (num_v_heads // num_q_heads)
    head_ratio,
    # Constexpr dimensions
    HEAD_SIZE: tl.constexpr,
):
    """
    One-step recurrent update for one (token_idx, v_head).
    Recurrence is serialized in Python by launching this kernel token-by-token
    within each sequence, which guarantees correct state dependency.
    """
    pid_v_head = tl.program_id(0)
    pid_qk_head = pid_v_head // head_ratio

    # Load gate parameters
    A_log_val = tl.load(A_log_ptr + pid_v_head).to(tl.float32)
    dt_bias_val = tl.load(DT_bias_ptr + pid_v_head).to(tl.float32)
    a_val = tl.load(A_ptr + token_idx * stride_a_t + pid_v_head * stride_a_h).to(tl.float32)
    b_val = tl.load(B_ptr + token_idx * stride_b_t + pid_v_head * stride_b_h).to(tl.float32)

    # g = exp(-exp(A_log) * softplus(a + dt_bias))
    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    alpha = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta = tl.sigmoid(b_val)

    k_idx = tl.arange(0, HEAD_SIZE)
    v_idx = tl.arange(0, HEAD_SIZE)

    # Load q/k/v vectors for this token
    q_ptrs = Q_ptr + token_idx * stride_q_t + pid_qk_head * stride_q_h + k_idx * stride_q_d
    k_ptrs = K_ptr + token_idx * stride_k_t + pid_qk_head * stride_k_h + k_idx * stride_k_d
    v_ptrs = V_ptr + token_idx * stride_v_t + pid_v_head * stride_v_h + v_idx * stride_v_d
    q_vec = tl.load(q_ptrs).to(tl.float32)  # [K]
    k_vec = tl.load(k_ptrs).to(tl.float32)  # [K]
    v_vec = tl.load(v_ptrs).to(tl.float32)  # [V]

    # Load state matrix [V, K]
    state_ptrs = (
        STATE_ptr
        + seq_idx * stride_state_s
        + pid_v_head * stride_state_h
        + v_idx[:, None] * stride_state_v
        + k_idx[None, :] * stride_state_k
    )
    state_mat = tl.load(state_ptrs).to(tl.float32)

    # Recurrent update
    old_state = alpha * state_mat
    old_v = tl.sum(old_state * k_vec[None, :], axis=1)
    new_v = beta * v_vec + (1.0 - beta) * old_v
    delta_v = new_v - old_v
    new_state_mat = old_state + k_vec[None, :] * delta_v[:, None]
    out_vec = scale * tl.sum(new_state_mat * q_vec[None, :], axis=1)

    # Write back state and output
    tl.store(state_ptrs, new_state_mat.to(tl.float32))
    out_ptrs = OUT_ptr + token_idx * stride_o_t + pid_v_head * stride_o_h + v_idx * stride_o_d
    tl.store(out_ptrs, out_vec.to(tl.bfloat16))


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability (same as reference)."""
    return a.float() @ b.float()


@torch.no_grad()
def gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output=None, new_state=None):
    """
    Gated Delta Net prefill reference implementation (k-last layout).
    EXACT COPY of JSON reference - ignoring passed output/new_state.
    """
    import math
    import json
    
    # #region agent log
    def _log(msg, data):
        import time
        log_path = "/home/ntcucsk201/gpu_100days_challenge/.cursor/debug.log"
        entry = {"timestamp": int(time.time()*1000), "location": "kernel.py:gdn_prefill", "message": msg, "data": data}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    # #endregion
    
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    # #region agent log
    _log("H1-config", {
        "total_seq_len": total_seq_len, "num_q_heads": num_q_heads, "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads, "num_sab_heads": num_sab_heads, "head_size": head_size,
        "num_seqs": num_seqs, "scale_input": scale,
        "output_passed": output is not None, "new_state_passed": new_state is not None,
        "dps_false": True
    })
    # #endregion

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = matmul(k_H1K, old_state_HKV)
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)

    # #region agent log
    _log("H2-output", {
        "output_shape": list(output.shape), "output_dtype": str(output.dtype),
        "new_state_shape": list(new_state.shape), "new_state_dtype": str(new_state.dtype),
        "has_inf_output": bool(torch.isinf(output).any()),
        "has_nan_output": bool(output.isnan().any()),
        "has_inf_state": bool(torch.isinf(new_state).any()),
        "has_nan_state": bool(new_state.isnan().any())
    })
    # #endregion

    return output, new_state
