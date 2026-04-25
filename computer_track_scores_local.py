"""
Compute per-track speedup scores for Wombat TW — pure-Python variant.

Reads ./mlsys26-contest/traces/**/*.jsonl directly and computes scores
without importing flashinfer_bench. Useful when the package is not
installable locally (e.g. macOS ARM cannot install `flashinfer`).

Scoring mirrors EVALUATION.md:
  1. Per-kernel speedup = arithmetic mean of per-workload
     (FlashInfer_baseline_latency / your_kernel_latency).
     Correctness-gated: any failing workload zeros the whole kernel.
  2. Per-track speedup = sum(per-kernel speedups) / expected_kernel_count.
     Missing/failing kernels contribute 0 to the numerator.

Usage:
  python compute_track_scores_local.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

DATASET_PATH = Path("./mlsys26-contest")
AUTHOR = "Wombat TW"
BASELINE_AUTHOR = "flashinfer"

# (op_type subdir name under traces/, expected_kernel_count for the track)
TRACKS = {
    "MoE": ("moe",       1),
    "DSA": ("dsa_paged", 2),
    "GDN": ("gdn",       2),
}

# Wombat TW's solution manifests live outside the dataset — include them
# so the author for our solution names can still be resolved.
EXTRA_SOLUTION_PATHS = [
    Path("./gdn_decode/solution.json"),
    Path("./gdn_prefill/solution.json"),
]


def build_author_map() -> Dict[str, str]:
    """solution_name -> normalized author."""
    m: Dict[str, str] = {}
    candidates = list((DATASET_PATH / "solutions").rglob("*.json")) + EXTRA_SOLUTION_PATHS
    for sol_path in candidates:
        if not sol_path.exists():
            continue
        try:
            obj = json.loads(sol_path.read_text())
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        author = (obj.get("author") or "").strip()
        if not name:
            continue
        # Normalize baseline authors (e.g. "flashinfer, deep_gemm" -> "flashinfer")
        if author.startswith("flashinfer"):
            author = "flashinfer"
        m[name] = author
    return m


def load_traces() -> List[Tuple[str, dict]]:
    """Return list of (op_type, trace_dict). op_type is inferred from path."""
    traces_root = DATASET_PATH / "traces"
    out: List[Tuple[str, dict]] = []
    if not traces_root.exists():
        return out
    for jsonl in traces_root.rglob("*.jsonl"):
        op_type = jsonl.parent.name
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append((op_type, json.loads(line)))
                except json.JSONDecodeError:
                    continue
    return out


def latest_per_workload(
    records: List[Tuple[str, dict]],
) -> Dict[Tuple[str, str, str], Tuple[str, dict]]:
    """Dedupe by (definition, solution_name, workload_uuid). Keep latest timestamp."""
    latest: Dict[Tuple[str, str, str], Tuple[str, dict]] = {}
    for op_type, t in records:
        key = (
            t.get("definition"),
            t.get("solution"),
            (t.get("workload") or {}).get("uuid"),
        )
        if None in key:
            continue
        ts_new = (t.get("evaluation") or {}).get("timestamp", "")
        existing = latest.get(key)
        ts_old = (existing[1].get("evaluation") or {}).get("timestamp", "") if existing else ""
        if existing is None or ts_new >= ts_old:
            latest[key] = (op_type, t)
    return latest


def per_kernel_speedup(
    ours: Dict[str, dict],
    baseline: Dict[str, dict],
) -> Tuple[float, int, int, int, int]:
    """Compute per-kernel speedup with correctness gate.

    For each workload, per-workload speedup is (baseline_latency / our_latency).
    When no separate baseline trace is available, fall back to the in-run
    `reference_latency_ms` recorded on our own trace (same GPU, same workload),
    which is equivalent to `speedup_factor` on that trace.

    Returns
    -------
    (speedup, n_scored_from_baseline, n_scored_from_reference, n_failed, n_skipped)
    """
    speedups: List[float] = []
    n_from_baseline = 0
    n_from_reference = 0
    n_failed = 0
    n_skipped = 0

    for wl_uuid, our_trace in ours.items():
        ev = our_trace.get("evaluation") or {}
        if ev.get("status") != "PASSED":
            n_failed += 1
            continue
        our_lat = (ev.get("performance") or {}).get("latency_ms")
        if not our_lat:
            n_skipped += 1
            continue

        base = baseline.get(wl_uuid)
        base_ev = (base or {}).get("evaluation") or {}
        base_lat = (base_ev.get("performance") or {}).get("latency_ms")

        if base and base_ev.get("status") == "PASSED" and base_lat:
            speedups.append(base_lat / our_lat)
            n_from_baseline += 1
            continue

        # Fallback: use in-run reference latency from our own trace.
        ref_lat = (ev.get("performance") or {}).get("reference_latency_ms")
        if ref_lat:
            speedups.append(ref_lat / our_lat)
            n_from_reference += 1
            continue

        n_skipped += 1

    if n_failed > 0:  # correctness gate
        return 0.0, n_from_baseline, n_from_reference, n_failed, n_skipped
    if not speedups:
        return 0.0, 0, 0, 0, n_skipped
    return (
        sum(speedups) / len(speedups),
        n_from_baseline,
        n_from_reference,
        0,
        n_skipped,
    )


def main() -> None:
    author_map = build_author_map()
    records = load_traces()
    latest = latest_per_workload(records)

    # (definition, author) -> {workload_uuid: trace}
    by_def_author: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)
    # definition -> op_type
    def_to_op: Dict[str, str] = {}

    for (def_name, sol_name, wl_uuid), (op_type, t) in latest.items():
        author = author_map.get(sol_name)
        if not author:
            continue
        by_def_author[(def_name, author)][wl_uuid] = t
        def_to_op[def_name] = op_type

    print(f"Author:   {AUTHOR!r}")
    print(f"Baseline: {BASELINE_AUTHOR!r}")
    print(f"Dataset:  {DATASET_PATH}")
    print()

    for track, (op_type, expected) in TRACKS.items():
        defs_in_track = sorted({d for d, op in def_to_op.items() if op == op_type})
        our_defs = [d for d in defs_in_track if (d, AUTHOR) in by_def_author]

        print(f"{track} ({op_type}, expected {expected} kernel(s)):")

        if not our_defs:
            print(f"  no traces for author={AUTHOR!r}")
            print(f"  -> Track speedup: 0.000x  (missing-kernel halving: 0/{expected})")
            print()
            continue

        kernel_speedups: List[float] = []
        for d in our_defs:
            ours = by_def_author[(d, AUTHOR)]
            baseline = by_def_author.get((d, BASELINE_AUTHOR), {})
            sp, n_base, n_ref, n_fail, n_skip = per_kernel_speedup(ours, baseline)

            n_ok = n_base + n_ref
            tag = ""
            if n_fail > 0:
                tag = f"  (CORRECTNESS-GATED: {n_fail} failed, {n_ok} passed)"
            parts = []
            if n_base:
                parts.append(f"{n_base} vs baseline trace")
            if n_ref:
                parts.append(f"{n_ref} vs in-run reference_latency")
            if n_skip:
                parts.append(f"{n_skip} skipped")
            detail = "[" + ", ".join(parts) + "]" if parts else "[no scored workloads]"
            print(f"  {d}: {sp:.3f}x  {detail}{tag}")
            kernel_speedups.append(sp)

        track_speedup = sum(kernel_speedups) / expected
        halved = (
            ""
            if len(kernel_speedups) == expected
            else f"  (missing-kernel halving: {len(kernel_speedups)}/{expected})"
        )
        print(f"  -> Track speedup: {track_speedup:.3f}x{halved}")
        print()


if __name__ == "__main__":
    main()
