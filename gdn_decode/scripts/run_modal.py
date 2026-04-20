"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
# Volume always mounted at /data. Trace set root: /data/mlsys26-contest if you put parent dir,
# or /data if you put mlsys26-contest contents directly at volume root.
TRACE_SET_PATH = "/data/mlsys26-contest"
VOLUME_MOUNT_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

# B200 first for benchmark; fallback to H100/A100 if B200 unavailable (e.g. Triton compat)
GPU_FALLBACK = ["B200", "H100", "A100"]


@app.function(image=image, gpu=GPU_FALLBACK, timeout=3600, volumes={VOLUME_MOUNT_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    # Try trace set path; fallback to volume root if empty (e.g. put mlsys26-contest contents at root)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if not trace_set.definitions and TRACE_SET_PATH != VOLUME_MOUNT_PATH:
        trace_set = TraceSet.from_path(VOLUME_MOUNT_PATH)

    if solution.definition not in trace_set.definitions:
        available = sorted(trace_set.definitions.keys()) if trace_set.definitions else []
        hint = (
            "Trace set is empty. Upload the mlsys26-contest dataset to the Modal volume:\n"
            "  modal volume put flashinfer-trace /path/to/mlsys26-contest\n"
            "Use the directory containing definitions/ and workloads/ (e.g. mlsys26-contest clone)."
        ) if not available else (
            f"Available definitions: {available[:20]}{'...' if len(available) > 20 else ''}"
        )
        raise ValueError(
            f"Definition '{solution.definition}' not found in trace set. {hint}"
        )

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
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


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


def save_results(results: dict, solution_name: str, definition: str):
    """Save benchmark results to a JSON file."""
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PROJECT_ROOT / "results" / f"{definition}_{solution_name}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "solution_name": solution_name,
        "definition": definition,
        "timestamp": timestamp,
        "gpu": "B200 (Modal)",
        "benchmark_config": {
            "warmup_runs": 3,
            "iterations": 100,
            "num_trials": 5,
        },
        "results": results,
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
    save_results(results, solution.name, solution.definition)