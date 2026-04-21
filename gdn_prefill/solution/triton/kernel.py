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

@triton.jit
def gdn_prefill_chunk_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr,
    STATE_ptr,              # in/out: [num_seqs, num_v_heads, V, K]
    A_log_ptr, A_ptr, DT_bias_ptr, B_ptr,
    # Output pointer
    OUT_ptr,                # [total_seq_len, num_v_heads, head_size]
    # Sequence boundaries
    CU_SEQLENS_ptr,
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
    scale,
    # Constexpr dimensions
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
):
    """
    Chunk-wise Gated Delta Rule prefill kernel.

    Within each chunk of BLOCK_T tokens we re-formulate the recurrent
    state update into matrix-matrix operations that use Tensor Cores:

        S_t = gamma_t * S_0 + sum_{j<=t} (gamma_t/gamma_j) * U'_j * k_j^T
        (I + A') U' = rhs
          where A'_{t,j} = beta_t * (gamma_t/gamma_j) * (k_j . k_t)   (j < t)
          and   rhs_t   = beta_t * (v_t - gamma_t * S_0 k_t)

    All GEMMs run on Tensor Cores. The only per-token sequential part is
    the BLOCK_T-step forward substitution, which is cheap for BLOCK_T=16.

    Outputs:
        O_t = scale * (gamma_t * (S_0 q_t) + sum_{j<=t} (gamma_t/gamma_j) (q_t.k_j) U'_j)
    State update:
        S_new = G * S_0 + W^T K      where W_t = (G/gamma_t) * U'_t
                                            G = gamma_{BLOCK_T-1}
    """
    pid = tl.program_id(0)
    pid_v_tile = pid % (HEAD_SIZE // BLOCK_V)
    tmp = pid // (HEAD_SIZE // BLOCK_V)
    pid_v_head = tmp % NUM_V_HEADS
    pid_seq = tmp // NUM_V_HEADS
    pid_qk_head = pid_v_head // HEAD_RATIO

    seq_start = tl.load(CU_SEQLENS_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(CU_SEQLENS_ptr + pid_seq + 1).to(tl.int32)
    seq_len = seq_end - seq_start

    # ---- Invariant loads ----
    A_log_val = tl.load(A_log_ptr + pid_v_head).to(tl.float32)
    dt_bias_val = tl.load(DT_bias_ptr + pid_v_head).to(tl.float32)
    neg_exp_A_log = -tl.exp(A_log_val)

    # ---- Offset vectors ----
    v_off = pid_v_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_off < HEAD_SIZE
    k_idx = tl.arange(0, HEAD_SIZE)
    t_idx = tl.arange(0, BLOCK_T)

    tl.multiple_of(k_idx, 16)
    tl.multiple_of(v_off, 16)

    # ---- Causal mask matrices (compile-time) ----
    strict_lower = t_idx[:, None] > t_idx[None, :]    # j < t
    tril_incl = t_idx[:, None] >= t_idx[None, :]      # j <= t

    # ---- Load initial state tile S_0: [BLOCK_V, HEAD_SIZE] ----
    state_block_ptr = tl.make_block_ptr(
        base=STATE_ptr + pid_seq * stride_state_s + pid_v_head * stride_state_h,
        shape=(HEAD_SIZE, HEAD_SIZE),
        strides=(stride_state_v, stride_state_k),
        offsets=(pid_v_tile * BLOCK_V, 0),
        block_shape=(BLOCK_V, HEAD_SIZE),
        order=(1, 0),
    )
    S = tl.load(state_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    last_mask_f = (t_idx == (BLOCK_T - 1)).to(tl.float32)

    # ---- Main chunk loop ----
    num_chunks = tl.cdiv(seq_len, BLOCK_T)
    chunk_idx = 0
    while chunk_idx < num_chunks:
        chunk_base = chunk_idx * BLOCK_T
        valid_mask = (chunk_base + t_idx) < seq_len             # [BLOCK_T]
        tok_idx = seq_start + chunk_base + t_idx                # [BLOCK_T]

        # ---- Load Q, K: [BLOCK_T, HEAD_SIZE] in bf16 for HMMA.BF16.F32 ----
        # Keeping bf16 enables faster Tensor Core ops (HMMA.BF16) vs TF32.
        # Trade-off: bf16 has 7-bit mantissa vs TF32's 10-bit, but for Q/K
        # attention-like ops this is typically acceptable.
        q_ptrs = (Q_ptr + tok_idx[:, None] * stride_q_t
                  + pid_qk_head * stride_q_h + k_idx[None, :] * stride_q_d)
        Q_bf16 = tl.load(q_ptrs, mask=valid_mask[:, None], other=0.0).to(tl.bfloat16)
        Q_fp32 = Q_bf16.to(tl.float32)  # For scalar ops that need fp32

        k_ptrs = (K_ptr + tok_idx[:, None] * stride_k_t
                  + pid_qk_head * stride_k_h + k_idx[None, :] * stride_k_d)
        K_bf16 = tl.load(k_ptrs, mask=valid_mask[:, None], other=0.0).to(tl.bfloat16)
        K_fp32 = K_bf16.to(tl.float32)  # For scalar ops that need fp32

        # ---- Load V: [BLOCK_T, BLOCK_V] ----
        v_ptrs = (V_ptr + tok_idx[:, None] * stride_v_t
                  + pid_v_head * stride_v_h + v_off[None, :] * stride_v_d)
        V = tl.load(v_ptrs, mask=valid_mask[:, None] & v_mask[None, :], other=0.0).to(tl.float32)

        # ---- Load a, b: [BLOCK_T] ----
        a_vec = tl.load(A_ptr + tok_idx * stride_a_t + pid_v_head * stride_a_h,
                        mask=valid_mask, other=0.0).to(tl.float32)
        b_vec = tl.load(B_ptr + tok_idx * stride_b_t + pid_v_head * stride_b_h,
                        mask=valid_mask, other=0.0).to(tl.float32)

        # ---- alpha, beta per token (valid tokens only) ----
        x = a_vec + dt_bias_val
        softplus_x = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        log_alpha = neg_exp_A_log * softplus_x
        log_alpha = tl.maximum(log_alpha, -88.0)
        # Invalid tokens: log_alpha = 0 (alpha=1), beta = 0
        log_alpha = tl.where(valid_mask, log_alpha, 0.0)
        beta = tl.where(valid_mask, tl.sigmoid(b_vec), 0.0)

        # ---- Cumulative decay: log_gamma[t] = sum log_alpha[0..t] ----
        log_gamma = tl.cumsum(log_alpha, axis=0)
        log_gamma = tl.maximum(tl.minimum(log_gamma, 88.0), -88.0)  # Bidirectional clamp
        gamma = tl.exp(log_gamma)                                # [BLOCK_T]

        # G = gamma[BLOCK_T - 1]
        G = tl.sum(gamma * last_mask_f)
        log_G = tl.sum(log_gamma * last_mask_f)

        # ---- K @ S_0^T: [BLOCK_T, BLOCK_V] (bf16 @ bf16 -> fp32 via HMMA.BF16) ----
        # Convert S to bf16 for Tensor Core; output accumulates in fp32.
        S_bf16 = S.to(tl.bfloat16)
        KS0T = tl.dot(K_bf16, tl.trans(S_bf16))  # HMMA.BF16.F32

        # rhs[t] = beta[t] * (V[t] - gamma[t] * KS0T[t])
        rhs = beta[:, None] * (V - gamma[:, None] * KS0T)

        # ---- K @ K^T: [BLOCK_T, BLOCK_T] (bf16 @ bf16 -> fp32 via HMMA.BF16) ----
        KKT = tl.dot(K_bf16, tl.trans(K_bf16))  # HMMA.BF16.F32

        # gamma_ratio[t, j] = gamma[t] / gamma[j] = exp(log_gamma[t] - log_gamma[j])
        # Bidirectional clamp to avoid overflow/underflow in exp
        dlog = log_gamma[:, None] - log_gamma[None, :]
        dlog = tl.maximum(tl.minimum(dlog, 88.0), -88.0)
        gamma_ratio = tl.exp(dlog)

        # A'[t, j] = beta[t] * gamma_ratio[t, j] * KKT[t, j]  for j < t, else 0
        A_mat = tl.where(strict_lower, beta[:, None] * gamma_ratio * KKT, 0.0)

        # ---- Forward substitution: (I + A') U' = rhs ----
        U = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
        for t_step in tl.static_range(BLOCK_T):
            row_bool = t_idx == t_step
            row_f = row_bool.to(tl.float32)
            A_row = tl.sum(A_mat * row_f[:, None], axis=0)                 # [BLOCK_T]
            rhs_row = tl.sum(rhs * row_f[:, None], axis=0)                 # [BLOCK_V]
            contrib = tl.sum(A_row[:, None] * U, axis=0)                   # [BLOCK_V]
            U_new = rhs_row - contrib
            U = tl.where(row_bool[:, None], U_new[None, :], U)

        # ---- Output: scale * (gamma * (Q @ S_0^T) + B @ U') ----
        # Q @ S^T: bf16 @ bf16 -> fp32 via HMMA.BF16
        Out_inter = tl.dot(Q_bf16, tl.trans(S_bf16))                       # [BLOCK_T, BLOCK_V]
        # Q @ K^T: bf16 @ bf16 -> fp32 via HMMA.BF16
        QKT = tl.dot(Q_bf16, tl.trans(K_bf16))                             # [BLOCK_T, BLOCK_T]
        B_mat = tl.where(tril_incl, gamma_ratio * QKT, 0.0)                # [BLOCK_T, BLOCK_T]
        # B_mat @ U: B_mat is fp32, U is fp32 - convert to bf16 for TC
        Out_intra = tl.dot(B_mat.to(tl.bfloat16), U.to(tl.bfloat16))       # [BLOCK_T, BLOCK_V]
        Out = scale * (gamma[:, None] * Out_inter + Out_intra)

        out_ptrs = (OUT_ptr + tok_idx[:, None] * stride_o_t
                    + pid_v_head * stride_o_h + v_off[None, :] * stride_o_d)
        tl.store(out_ptrs, Out.to(tl.bfloat16),
                 mask=valid_mask[:, None] & v_mask[None, :])

        # ---- State update: S_new = G * S_0 + W^T @ K ----
        # W[t] = (G / gamma[t]) * U'[t] = exp(log_G - log_gamma[t]) * U'[t]
        dlog_end = log_G - log_gamma
        dlog_end = tl.maximum(tl.minimum(dlog_end, 88.0), -88.0)           # Bidirectional clamp
        G_over_gamma = tl.exp(dlog_end)                                    # [BLOCK_T]
        W = G_over_gamma[:, None] * U                                      # [BLOCK_T, BLOCK_V]
        # W^T @ K: bf16 @ bf16 -> fp32 via HMMA.BF16
        dS = tl.dot(tl.trans(W.to(tl.bfloat16)), K_bf16)                   # [BLOCK_V, HEAD_SIZE]
        S = G * S + dS

        chunk_idx += 1

    # Store final state
    tl.store(state_block_ptr, S.to(tl.float32), boundary_check=(0,))


@torch.no_grad()
def gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output=None, new_state=None):
    """
    Gated Delta Net prefill with sequential linear attention.

    Single kernel launch processes all sequences. V-dimension is tiled
    (BLOCK_V) so multiple thread blocks cover one (seq, v_head) in parallel.
    Tokens are processed sequentially via while loop within each program.
    """
    import math

    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    head_ratio = num_v_heads // num_q_heads

    if output is None:
        output = torch.zeros(
            (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
        )
    else:
        output.zero_()

    if new_state is None:
        if state is None:
            new_state = torch.zeros(
                (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )
        else:
            new_state = state.clone()
    else:
        if state is None:
            new_state.zero_()
        else:
            new_state.copy_(state)

    if num_seqs == 0 or total_seq_len == 0:
        return output, new_state

    # For the chunk-wise kernel we need more tile tensors alive simultaneously
    # (Q/K chunk, U', A', gamma_ratio, ...), so go back to BLOCK_V=32 to keep
    # register + shared memory pressure manageable. State tile is 32x128 fp32
    # = 16KB, still large enough to benefit from TMA on Hopper.
    BLOCK_V = 32
    BLOCK_T = 16  # Chunk length; must be a compile-time constant, >=16 for TC.
    num_v_tiles = head_size // BLOCK_V
    grid = (num_seqs * num_v_heads * num_v_tiles,)

    gdn_prefill_chunk_kernel[grid](
        q, k, v,
        new_state,
        A_log, a, dt_bias, b,
        output,
        cu_seqlens,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        scale,
        HEAD_SIZE=head_size,
        BLOCK_V=BLOCK_V,
        BLOCK_T=BLOCK_T,
        NUM_V_HEADS=num_v_heads,
        HEAD_RATIO=head_ratio,
        num_warps=4,
        num_stages=2,
    )

    return output, new_state
