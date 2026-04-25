"""
Compute per-track speedup scores for Wombat TW locally.

Mirrors the official flashinfer-bench evaluation pipeline's
compute_track_scores.py. Reads traces from ./mlsys26-contest and prints
each track's speedup after missing-kernel halving.

Prerequisites:
  - ./mlsys26-contest/traces/ exists and has recent traces
    (pull from Modal volume after run_modal.py finishes).
  - ./mlsys26-contest/solutions/ contains Wombat TW's solution JSONs
    alongside the baselines.

Usage:
  python compute_track_scores.py
"""

from flashinfer_bench.data import TraceSet

DATASET_PATH = "./mlsys26-contest"
AUTHOR = "Wombat TW"
BASELINE_AUTHOR = "flashinfer"

TRACKS = {
    "MoE": ("moe",       1),
    "DSA": ("dsa_paged", 2),
    "GDN": ("gdn",       2),
}


def main() -> None:
    trace_set = TraceSet.from_path(DATASET_PATH)

    # Normalize baseline author — DSA indexer baseline carries
    # a combined author "flashinfer, deep_gemm" that we collapse to "flashinfer".
    for sols in trace_set.solutions.values():
        for i, sol in enumerate(sols):
            if sol.author and sol.author.startswith("flashinfer") and sol.author != "flashinfer":
                sols[i] = sol.model_copy(update={"author": "flashinfer"})
                trace_set._solution_by_name[sol.name] = sols[i]

    print(f"Author:   {AUTHOR!r}")
    print(f"Baseline: {BASELINE_AUTHOR!r}")
    print(f"Dataset:  {DATASET_PATH}")
    print()

    any_result = False
    for track, (op_type, expected) in TRACKS.items():
        s = trace_set.get_author_score(
            AUTHOR,
            baseline_author=BASELINE_AUTHOR,
            op_type=op_type,
        )
        if s is None:
            print(f"{track}: (no score — no matching traces found)")
            continue
        any_result = True
        track_speedup = s.avg_speedup * s.definitions / expected
        print(f"{track}: {track_speedup:.3f}x  ({s.definitions}/{expected} kernels)")

    if not any_result:
        print()
        print("No scores produced. Check that:")
        print(f"  1. {DATASET_PATH}/traces/ exists and has recent traces.")
        print(f"  2. {DATASET_PATH}/solutions/ has a solution with author={AUTHOR!r}.")


if __name__ == "__main__":
    main()