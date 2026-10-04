"""Collect explicitly selected completed runs into a small shareable result bundle."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("results/initial"))
    parser.add_argument("--report", type=Path, default=Path("docs/INITIAL_RESULTS.md"))
    args = parser.parse_args()
    results = [json.loads((p/"results.json").read_text(encoding="utf-8")) for p in args.runs]
    if len({r["config"]["seed"] for r in results}) != len(results):
        raise ValueError("Duplicate seeds cannot count as independent runs")
    baseline = {k: v for k, v in results[0]["config"].items() if k != "seed"}
    if any({k: v for k, v in r["config"].items() if k != "seed"} != baseline for r in results):
        raise ValueError("Only runs with identical configuration apart from seed can be pooled")
    if any(r["completed_student_steps"] != r["config"]["student_steps"] for r in results):
        raise ValueError("Run did not finish the configured number of steps")
    args.output.mkdir(parents=True, exist_ok=True)
    for result in results:
        (args.output/f"seed-{result['config']['seed']}.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    source_files = sorted(Path("breadcrumb_memory").glob("*.py")) + sorted(Path("configs").glob("*.json"))
    manifest = {str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (args.output/"source-sha256.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    runtime = [r["elapsed_seconds"] for r in results]
    peak = [r["peak_cuda_allocated_bytes"]/(1024**2) for r in results]
    first = results[0]
    lines = ["# Initial measured results", "", "## Scope", "",
             f"Pilot across {len(results)} training seeds: {', '.join(str(r['config']['seed']) for r in results)}. "
             f"Each run used {first['config']['teacher_steps']} teacher steps, {first['config']['student_steps']} student steps, "
             f"and {first['config']['eval_examples']} held-out episodes per length/query condition.", "",
             f"Device: **{first['environment'].get('gpu', first['device'])}**. "
             f"Python {first['environment']['python']}; PyTorch {first['environment']['torch']}; CUDA runtime {first['environment']['cuda_runtime']}.",
             f"Teacher parameters: {first['teacher_parameters']:,}; student parameters: {first['student_parameters']:,}. "
             "This is a task-specific neural attention model, not a language model.", "",
             f"Measured run time: {min(runtime):.1f}–{max(runtime):.1f} seconds after process import, including evaluation. "
             f"Peak PyTorch allocated GPU tensors: {min(peak):.1f}–{max(peak):.1f} MiB. "
             "This excludes the CUDA context, allocator reservation and other applications; it is not total process VRAM.", "",
             "## Mean accuracy across training seeds", "",
             "| Query | Records | Teacher | Before training | After training | Disabled memory | Shuffled memory | Exact archive | Evidence retained |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, row in enumerate(first["after"]):
        group = [r["after"][i] for r in results]
        if any((x["length"], x["query_mode"]) != (row["length"], row["query_mode"]) for x in group):
            raise ValueError("Mismatched evaluation conditions")
        avg = lambda name: statistics.mean(x["accuracy"][name] for x in group)
        before = statistics.mean(r["before"][i]["accuracy"]["student"] for r in results)
        retained = statistics.mean(x["support_retained_fraction"] for x in group)
        lines.append(f"| {row['query_mode']} | {row['length']} | {avg('teacher'):.1%} | {before:.1%} | {avg('student'):.1%} | {avg('no_memory'):.1%} | {avg('shuffled'):.1%} | {avg('exact_archive'):.1%} | {retained:.1%} |")
    lines += ["", "## Per-seed old-query results at 40 records", "",
              "| Seed | Before | After | Improvement | Student KL |",
              "|---:|---:|---:|---:|---:|"]
    for r in results:
        condition = lambda rows: next(x for x in rows if x["query_mode"] == "old" and x["length"] == 40)
        if not any(x["length"] == 40 for x in r["after"]):
            continue
        before, after = condition(r["before"]), condition(r["after"])
        b, a = before["accuracy"]["student"], after["accuracy"]["student"]
        lines.append(f"| {r['config']['seed']} | {b:.1%} | {a:.1%} | {(a-b)*100:+.2f} pp | {after['teacher_student_kl']:.4f} |")
    lines += ["", "## What this supports", "",
              "The implemented memory carries information about old bindings, and training improves its feature geometry over random initialization. "
              "A control that removes or swaps the relevant memory loses that ability. "
              "This is a functional test of the prototype and evaluation, not a new result about frontier LLMs.", "",
              "## What failed or remains unproved", "",
              "- Random features already solve much of the task. Comparing only against memory-disabled performance would exaggerate the contribution of training.",
              "- Exact keyed lookup reaches 100%. Nothing here establishes superiority over ordinary retrieval.",
              "- Old memory interferes with recent queries; the memory-disabled reader is stronger on those. A gated local path is a candidate next test.",
              "- Beyond 40 records, old supporting evidence is progressively overwritten. The model does not recover evidence that was truly lost.",
              "- The 16,384-byte slow state is larger than the raw toy records. No real KV-cache compression ratio or end-to-end memory savings have been demonstrated.",
              "- No pretrained LLM, learned retrieval controller, second timescale, conflicting updates or real tool logs have been tested.", "",
              "## Reproduce", "", "```powershell"]
    for r in results:
        seed = r["config"]["seed"]
        lines.append(f".\\.venv\\Scripts\\python.exe -m breadcrumb_memory train --config configs/pilot.json --output runs/reproduce-{seed} --device cuda --seed {seed}")
    lines += ["```", "", "The committed JSON files contain complete configurations, conditional accuracies, KL, sampling intervals and environment metadata. "
              "Each run records source hashes at execution time. `source-sha256.json` records source contents at collection time; per-run git revisions are recorded when a commit existed. "
              "No pretrained weights, local checkpoints, authentication material or conversation transcript are included.", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
