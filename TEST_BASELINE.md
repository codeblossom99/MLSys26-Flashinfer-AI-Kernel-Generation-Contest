# TEST_BASELINE — Wombat TW 本地測分 Playbook

> **Scope:** 本文件記錄我們（Wombat TW）為 MLSys 2026 FlashInfer 競賽所建立的「在官方 CI 評分之前自己先測分」的完整流程。涵蓋 Modal 上跑 Wombat TW solution、跑 flashinfer baseline、同步 traces、本地算分。
>
> **Relationship to `EVALUATION.md`:** `EVALUATION.md` 是官方 CI 評分流程（B200 bare-metal + Docker image + `flashinfer-bench run` CLI）。本文件是我們在 **Modal（雲端 B200）+ macOS（無 CUDA）** 的組合下，用自己的 script 模擬同一套 scoring，本機就能看到分數。

---

## 1. Overview｜流程總覽

我們的測分流程分成兩個平行 workflow：

```
 Wombat TW 自己的 kernel ──► modal run run_modal.py ──┐
                                                       │
                                                       ▼
                                              Modal Volume
                                              /mlsys26-contest/
                                               └── traces/gdn/*.jsonl
                                                       │
 FlashInfer baseline (one-time) ─► modal run run_modal_baseline.py ─┘
                                                       │
                                                       ▼
                                         modal volume get → local
                                                       │
                                                       ▼
                                     python compute_track_scores_local.py
                                                       │
                                                       ▼
                                              GDN / DSA / MoE speedup
```

兩側都跑完、都有 traces → 本機算分 = 真實 `flashinfer_baseline_latency / our_latency`。只跑 Wombat TW 不跑 baseline → fallback 到 `reference_latency_ms`（PyTorch naive ref），分數會**極度失真**（可以差 100x 以上）。

---

## 2. Prerequisites｜前置環境

### 2.1 Local machine（macOS / Linux）

```bash
# Python 3.12 + venv
python3.12 -m venv .venv
source .venv/bin/activate

# 核心套件（macOS 要用 --no-deps 繞過 nvidia-cudnn-frontend wheel 缺失）
pip install --no-deps flashinfer-bench
pip install pydantic                    # 補 pydantic 自己的 native deps
pip install --no-deps safetensors apache-tvm-ffi docstring-parser

# Modal CLI
pip install modal
modal setup                             # 一次性 token setup
```

> **Note (macOS):** `flashinfer-python`（CUDA-only）裝不起來，所以 `compute_track_scores.py`（用官方 `TraceSet.get_author_score` API）在 macOS 無法 run。我們用 `compute_track_scores_local.py` 純 Python parse jsonl 繞過。

### 2.2 Modal Volume（one-time）

```bash
# 建 volume（已建過可略）
modal volume create flashinfer-trace

# 上傳 contest dataset（clone from HuggingFace）
# 如果你還沒有本機 clone：
# huggingface-cli download flashinfer-ai/mlsys26-contest --repo-type dataset --local-dir mlsys26-contest

modal volume put flashinfer-trace ./mlsys26-contest /mlsys26-contest
```

Volume 結構驗證：

```bash
modal volume ls flashinfer-trace
# 應該看到 mlsys26-contest/ 底下有 definitions/ solutions/ workloads/ blob/ traces/
```

---

## 3. Directory Layout｜專案結構

```
mlsys-2026-flashInfer-c-wombat-tw/
├── EVALUATION.md                       # 官方評分規則（勿改）
├── TEST_BASELINE.md                    # 本文件
├── compute_track_scores.py             # 官方 API 版本（macOS 無法跑，留當 reference）
├── compute_track_scores_local.py       # 我們的 pure-Python scorer（任何機器可跑）
├── mlsys26-contest/                    # 官方 dataset 本地 clone（.gitignore）
│   ├── definitions/
│   ├── solutions/baseline/gdn/.../flashinfer_wrapper_*.json
│   ├── workloads/
│   ├── blob/
│   └── traces/gdn/*.jsonl              # 從 modal volume get 下來
├── gdn_decode/
│   ├── config.toml                     # solution metadata
│   ├── solution.json                   # 自動產生，git 可不追
│   ├── solution/triton/kernel.py       # ★ 我們的 kernel source
│   ├── scripts/
│   │   ├── pack_solution.py
│   │   ├── run_modal.py                # Wombat TW solution benchmark
│   │   └── run_modal_baseline.py       # flashinfer baseline benchmark
│   └── results/*.json                  # local-side summary（非必要）
└── gdn_prefill/                        # 結構同 gdn_decode
```

---

## 4. Iteration Loop｜Wombat TW Kernel 修改流程

改完 `gdn_decode/solution/triton/kernel.py` 或 `gdn_prefill/solution/triton/kernel.py` 之後：

```bash
# 1. Run benchmark on Modal B200 (auto-packs from source)
modal run gdn_decode/scripts/run_modal.py
modal run gdn_prefill/scripts/run_modal.py    # 只改了 decode 就不用跑 prefill
```

`run_modal.py` 的 `local_entrypoint` 會**自動 `pack_solution()`**，不用手動 pack。每次跑完 `gdn_<kernel>/solution.json` 會被覆寫（git 可能會看到 diff，不影響）。

```bash
# 2. Sync traces from Modal volume
rm -rf ./mlsys26-contest/traces            # 清舊的（不清會 Errno 21）
modal volume get flashinfer-trace /mlsys26-contest/traces ./mlsys26-contest/

# 3. Score
python compute_track_scores_local.py
```

**Expected output（有 baseline trace 時）：**

```
GDN (gdn, expected 2 kernel(s)):
  gdn_decode_qk4_v8_d128_k_last: 0.715x  [54 vs baseline trace]
  gdn_prefill_qk4_v8_d128_k_last: 1.668x  [100 vs baseline trace]
  -> Track speedup: 1.191x
```

關鍵是 detail 要寫 `[N vs baseline trace]`。如果寫的是 `[N vs in-run reference_latency]` → 代表 Volume 上沒有 flashinfer baseline 的 trace，分數失真，要先跑第 5 節的 baseline。

---

## 5. Baseline Setup (one-time)｜Baseline 跑法

**為什麼只跑一次：** flashinfer baseline 的 kernel 不會變，Volume 上的 baseline traces 寫一次就能重複用。只有 dataset 更新（新增 workload）時才需要重跑。

```bash
# Smoke test 先 (~30s for decode, ~20min for prefill first time including image build)
modal run gdn_decode/scripts/run_modal_baseline.py --smoke
modal run gdn_prefill/scripts/run_modal_baseline.py --smoke

# 確認 output 有 PASSED、abs_err 合理（< 1e-2）之後上 full
modal run gdn_decode/scripts/run_modal_baseline.py
modal run gdn_prefill/scripts/run_modal_baseline.py
```

Baseline full run 實測耗時：
- `gdn_decode`: ~10 min（54 workloads × 快 kernel）
- `gdn_prefill`: ~22 min（100 workloads × 慢 kernel）

第一次跑 image 要 build（裝 `flashinfer-python` + `flashinfer-cubin` ~1GB），後續 modal 會 cache。

---

## 6. Script Reference｜各 Script 用途

| Script | Purpose | Where it runs | Duration |
|---|---|---|---|
| `gdn_<kernel>/scripts/pack_solution.py` | 讀 `config.toml` + `kernel.py` 打包成 `solution.json` | Local | <1s |
| `gdn_<kernel>/scripts/run_modal.py` | Benchmark Wombat TW kernel on Modal B200（auto-pack） | Modal | decode ~10min / prefill ~30-60min |
| `gdn_<kernel>/scripts/run_modal_baseline.py` | Benchmark flashinfer baseline on Modal B200 | Modal | decode ~10min / prefill ~22min |
| `compute_track_scores.py` | 官方 API scorer（參考用） | Linux + CUDA env only | seconds |
| `compute_track_scores_local.py` | Pure-Python scorer（讀 jsonl） | Any Python 3 machine | <5s |

**CLI args：**

- `run_modal_baseline.py --smoke` — 只跑 1 workload + `warmup=1/iter=5/trials=1`，用來驗 image + GPU compat

---

## 7. Interpreting Scores｜看懂分數

### 7.1 Per-kernel speedup

```
per_kernel_speedup = mean(baseline_latency_ms / our_latency_ms)   # over workloads
```

- Correctness gate：只要 Wombat TW 有**任一** workload 不是 `PASSED`，整個 kernel 分數 → 0
- `[N vs baseline trace]` = 用 flashinfer baseline 的實測 latency 比較（**正確版**）
- `[N vs in-run reference_latency]` = fallback 用 trace 自己紀錄的 `reference_latency_ms`，那個是 **PyTorch naive reference**，不是 flashinfer baseline → 數字會誇大幾十到幾百倍

### 7.2 Per-track speedup

```
track_speedup = sum(per_kernel_speedups) / expected_kernel_count
```

- MoE expected = 1, DSA expected = 2, GDN expected = 2
- Missing kernel（沒跑過或歸 0）貢獻 0 給 numerator → single-kernel 投 GDN/DSA 等於被砍半

### 7.3 Known benchmarks

From `EVALUATION.md` announcement:
- `Agent-Assisted` leaderboard top: ~3.1x
- `Full-Agent` leaderboard top: ~1.2x

Wombat TW 目前 GDN track: 1.191x（decode 0.715x + prefill 1.668x）. Prefill 強、decode 弱。

---

## 8. Troubleshooting｜常見坑

### 8.1 `modal volume get ... [Errno 21] Is a directory`

**原因：** 本機 `./mlsys26-contest/traces/` 已存在（即使是空的）。

**解法：** `rm -rf ./mlsys26-contest/traces` 先清掉再 `modal volume get`.

---

### 8.2 `modal volume ls` 顯示舊時間，以為沒寫入

**現象：** 跑完 `modal run run_modal.py`，`modal volume ls flashinfer-trace /mlsys26-contest/traces` 仍顯示 Apr 17（dataset 上傳日），以為 write 失敗。

**原因：** `modal volume ls` 顯示的是 **directory entry mtime**，append 到裡面的 jsonl 不會更新 directory mtime。

**驗證：** 往下一層看 file-level mtime：

```bash
modal volume ls flashinfer-trace /mlsys26-contest/traces/gdn
# 看單一 .jsonl 檔的 mtime，那個才是真的更新時間
```

---

### 8.3 macOS 裝 `flashinfer-bench` 踩 `nvidia-cudnn-frontend` wheel 缺失

**解法：** 用 `--no-deps` 繞，並手動補 `pydantic`、`safetensors`、`apache-tvm-ffi`、`docstring-parser`。完整指令見 §2.1。

---

### 8.4 `AttributeError: 'TraceSet' object has no attribute 'get_author_score'`

**原因：** 裝的 `flashinfer-bench` 是 PyPI release 版本，`get_author_score` API 只在 main branch。

**解法：** 不升級（因為 main 需要 `flashinfer` which is CUDA-only）。改用 `compute_track_scores_local.py` 純 Python scorer。

---

### 8.5 `[N vs in-run reference_latency]` 出現但你期待 `[N vs baseline trace]`

**原因：** Modal Volume 上的 jsonl 沒有 flashinfer baseline 的 trace entry。

**解法：** 跑 §5 的 `run_modal_baseline.py` 然後重新 sync + score。

---

### 8.6 `reference_latency_ms` ≠ flashinfer baseline latency

**這是認知陷阱。** trace 裡的 `reference_latency_ms` 是 `definition.reference` 欄位（通常是 **naive PyTorch impl**），不是 `solutions/baseline/` 裡的 `flashinfer_wrapper_*` 的 latency。

驗證：看 source code `flashinfer_bench/compile/registry.py:build_reference()`：

```python
pseudo = Solution(
    name=f"{definition.name}__reference",
    ...
    sources=[SourceFile(path="main.py", content=definition.reference)],
)
```

`definition.reference` 是 definition YAML 裡的 reference 程式碼欄位，通常 PyTorch。

所以 `speedup_factor` in trace ≠ 官方 scoring。官方要跟 `flashinfer_wrapper_*` baseline 的 latency 比。

---

## 9. Teammates with Their Own GPU｜組員有自己的 GPU

例如組員有 **NVIDIA RTX 5090**（consumer Blackwell，SM120）：

### ✅ 可以做

- 本機跑 `flashinfer-bench run`（§EVALUATION.md 裡的 CLI）驗 correctness
- 本機用 `compute_track_scores_local.py` 算分（directional）
- Kernel iterate 快速驗 bug，不用等 Modal cold start

### ⚠️ 注意

- **5090 不是 B200**。SM120 vs SM100a、memory bandwidth、SM count 都不一樣 → latency 數字**僅供 directional 參考**，**不是官方分**
- `flashinfer-python` 在 5090 上（Blackwell consumer）能不能跑要實測，理論上 CUDA 13 + `flashinfer-cubin` 有 SM120 binary，但若沒有會退回 JIT 編譯，可能失敗
- 比賽 CI 只認 B200 結果。5090 上分數**高** ≠ 最終分數**高**（反之亦然）

### Setup（組員端）

```bash
# Linux + CUDA 13
pip install flashinfer-bench flashinfer-python flashinfer-cubin torch triton
git clone <our-repo>
cd mlsys-2026-flashInfer-c-wombat-tw

# Dataset（兩個方式擇一）
huggingface-cli download flashinfer-ai/mlsys26-contest --repo-type dataset --local-dir mlsys26-contest
# 或從隊友分享的 zip 解壓

# Run
flashinfer-bench run \
  --local ./mlsys26-contest \
  --definitions gdn_decode_qk4_v8_d128_k_last \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300

# Score
python compute_track_scores_local.py
```

**實務建議：** 組員用 5090 做 fast iteration（改 → 驗 → 看 rough perf），每隔幾次 iterate 再用 `modal run` 驗 B200 真實 latency。

---

## 10. Submission｜送件

送件 (commit / tag / push) 流程獨立寫在 [`SUBMISSION.md`](./SUBMISSION.md)。本檔只負責「測分」，不負責「送件」。

---

## Appendix A: Full End-to-End Test Run

從零開始到看到分數的完整 command sequence：

```bash
# 一次性 setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install --no-deps flashinfer-bench
pip install pydantic
pip install --no-deps safetensors apache-tvm-ffi docstring-parser
pip install modal
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace ./mlsys26-contest /mlsys26-contest

# Baseline (one-time)
modal run gdn_decode/scripts/run_modal_baseline.py
modal run gdn_prefill/scripts/run_modal_baseline.py

# 每次 iterate
modal run gdn_decode/scripts/run_modal.py
modal run gdn_prefill/scripts/run_modal.py
rm -rf ./mlsys26-contest/traces
modal volume get flashinfer-trace /mlsys26-contest/traces ./mlsys26-contest/
python compute_track_scores_local.py
```