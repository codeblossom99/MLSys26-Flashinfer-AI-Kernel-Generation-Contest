"""
FlashInfer-Bench Modal Baseline Benchmark Runner (gdn_prefill).

Benchmarks the FlashInfer baseline solution (`flashinfer_wrapper_123ca6`)
on the same Modal volume / workloads as run_modal.py, so that
compute_track_scores_local.py can score against a real baseline trace
instead of falling back to `reference_latency_ms` (which is the PyTorch
reference, not the optimized baseline).

Smoke test first (1 workload) to verify flashinfer-python + B200 compat:
    modal run gdn_prefill/scripts/run_modal_baseline.py --smoke

Full run:
    modal run gdn_prefill/scripts/run_modal_baseline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

app = modal.App("flashinfer-bench-baseline-gdn-prefill")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data/mlsys26-contest"
VOLUME_MOUNT_PATH = "/data"

DEFINITION_NAME = "gdn_prefill_qk4_v8_d128_k_last"
BASELINE_SOLUTION_NAME = "flashinfer_wrapper_123ca6"

# Same image stack as the decode baseline runner.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "flashinfer-bench",
        "flashinfer-python",
        "flashinfer-cubin",
        "torch",
        "triton",
        "numpy",
    )
)

# Prefer B200 to match run_modal.py; fall back to H200/H100 if baseline
# doesn't compile/run on Blackwell.
GPU_FALLBACK = ["B200", "H200", "H100"]


@app.function(
    image=image,
    gpu=GPU_FALLBACK,
    timeout=10800,
    volumes={VOLUME_MOUNT_PATH: trace_volume},
)
def run_baseline(smoke: bool = False) -> dict:
    """Run the flashinfer baseline on Modal and persist traces to the volume."""
    if smoke:
        config = BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=1)
    else:
        # Match run_modal.py's prefill config.
        config = BenchmarkConfig(warmup_runs=1, iterations=10, num_trials=2)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if not trace_set.definitions and TRACE_SET_PATH != VOLUME_MOUNT_PATH:
        trace_set = TraceSet.from_path(VOLUME_MOUNT_PATH)

    if DEFINITION_NAME not in trace_set.definitions:
        available = sorted(trace_set.definitions.keys())
        raise ValueError(
            f"Definition '{DEFINITION_NAME}' not found in trace set. "
            f"Available: {available[:20]}"
        )

    definition = trace_set.definitions[DEFINITION_NAME]
    all_solutions = trace_set.solutions.get(DEFINITION_NAME, [])
    baseline = next(
        (s for s in all_solutions if s.name == BASELINE_SOLUTION_NAME), None
    )
    if baseline is None:
        names = [s.name for s in all_solutions]
        raise ValueError(
            f"Baseline solution '{BASELINE_SOLUTION_NAME}' not found for "
            f"definition '{DEFINITION_NAME}'. Available: {names}"
        )

    workloads = trace_set.workloads.get(DEFINITION_NAME, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{DEFINITION_NAME}'")

    if smoke:
        workloads = workloads[:1]
        print(f"[smoke] Running against {len(workloads)} workload(s) only")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [baseline]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    trace_volume.commit()

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}
    for trace in traces:
        if not trace.evaluation:
            continue
        entry = {
            "status": trace.evaluation.status.value,
            "solution": trace.solution,
        }
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict) -> None:
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")
            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")
            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x vs reference", end="")
            if result.get("max_abs_error") is not None:
                print(
                    f" | abs_err={result['max_abs_error']:.2e}, "
                    f"rel_err={result.get('max_rel_error', 0):.2e}",
                    end="",
                )
            print()


def save_results(results: dict, smoke: bool) -> None:
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "smoke" if smoke else "full"
    output_path = (
        PROJECT_ROOT
        / "results"
        / f"{DEFINITION_NAME}_baseline_{BASELINE_SOLUTION_NAME}_{tag}_{timestamp}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "solution_name": BASELINE_SOLUTION_NAME,
        "definition": DEFINITION_NAME,
        "timestamp": timestamp,
        "gpu": "B200 (Modal, with H200/H100 fallback)",
        "smoke_test": smoke,
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


@app.local_entrypoint()
def main(smoke: bool = False):
    """Benchmark the flashinfer baseline on Modal."""
    print(
        f"Running baseline '{BASELINE_SOLUTION_NAME}' on definition "
        f"'{DEFINITION_NAME}' (smoke={smoke})..."
    )
    results = run_baseline.remote(smoke=smoke)
    if not results:
        print("No results returned!")
        return
    print_results(results)
    save_results(results, smoke)
