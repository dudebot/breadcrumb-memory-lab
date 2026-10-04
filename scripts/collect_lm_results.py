"""Publish small, explicitly selected LM experiment reports, never weights."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--probe", type=Path)
    args = parser.parse_args()
    results = [json.loads((path/"results.json").read_text(encoding="utf-8")) for path in args.runs]
    if len({r["seed"] for r in results}) != len(results):
        raise ValueError("Duplicate seed")
    if any(not r["frozen_backbone_verified"] or r["steps_completed"] != r["steps_requested"] for r in results):
        raise ValueError("Only completed runs with frozen backbones may be collected")
    for key in ("model", "revision", "steps_completed", "batch_size", "eval_examples"):
        if any(r[key] != results[0][key] for r in results):
            raise ValueError(f"Cannot combine runs with different {key}")
    out = Path("results/lm")
    out.mkdir(parents=True, exist_ok=True)
    for result in results:
        (out/f"seed-{result['seed']}.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    if args.probe:
        (out/"probe-135m.json").write_text(args.probe.read_text(encoding="utf-8"), encoding="utf-8")
    first = results[0]
    lines = ["# Small language-model experiment", "",
             "Accuracy on this task alone does not establish object/color binding. "
             "See [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md) for the stricter quartet "
             "evaluation of the original saved adapters and its negative result.", "",
             "## What was run", "",
             f"Backbone: `{first['model']}`, pinned to `{first['revision']}`. "
             f"All {first['backbone_parameters']:,} backbone parameters were frozen. "
             f"Only a {first['trainable_parameters']:,}-parameter compressor was trained.", "",
             "Four factual sentences assign random colors to objects. The teacher sees all four. The student sees the last two "
             "plus four learned vectors encoding the first two. Those vectors are created before the query is supplied to the reader. "
             "The task is to predict the next color token in a fixed sentence-completion format.", "",
             "Training uses full-vocabulary teacher/student KL, with no extra ground-truth answer loss. "
             "This is a single-prefix compression test. It has no recurrent ring, learned eviction or archive retrieval.", "",
             f"Each run completed {first['steps_completed']:,} updates at batch size {first['batch_size']}: "
             f"{first['steps_completed']*first['batch_size']:,} freshly randomized training episodes. "
             f"Evaluation uses {first['eval_examples']} independently generated examples per condition, the same before and after training. "
             "Training and evaluation share objects, colors and sentence templates; this does not test unseen language formats.", "",
             "## Four-fact accuracy: actual highest-probability next token", "",
             "| Seed | Full text | Recent text only | Untrained memory | Trained memory | Zero vectors | Other episode's memory |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        a, b = r["after"]["metrics"], r["before"]["metrics"]
        score = lambda name: a[name]["raw_next_token_accuracy"]
        lines.append(f"| {r['seed']} | {score('teacher'):.1%} | {score('local'):.1%} | {b['student']['raw_next_token_accuracy']:.1%} | {score('student'):.1%} | {score('zero_memory'):.1%} | {score('shuffled_memory'):.1%} |")
    lines += ["", "JSON additionally reports forced choice among eight color tokens. The table above uses the whole vocabulary. "
              "A uniform guess among the eight colors would be correct 12.5% of the time; raw next-token output is not constrained to those colors.", "",
              "## Eight-fact extrapolation", "",
              "| Seed | Full text | Recent only | Trained memory | Shuffled memory |",
              "|---:|---:|---:|---:|---:|"]
    for r in results:
        a = r["longer_history"]["metrics"]
        score = lambda name: a[name]["raw_next_token_accuracy"]
        lines.append(f"| {r['seed']} | {score('teacher'):.1%} | {score('local'):.1%} | {score('student'):.1%} | {score('shuffled_memory'):.1%} |")
    lines += ["", "The teacher itself is considerably weaker with eight facts. The long-history scores cannot establish reliable long-context capability.", "",
              "## Resource measurements", "",
              f"GPU: {first['gpu']}; PyTorch {first['torch']}. All model computation used FP32. "
              "Times include model loading from the local cache, training and all reported evaluation controls, but exclude the initial model download.", "",
              "| Seed | Seconds | Peak allocated MiB | Peak reserved MiB | Teacher/student KL before | KL after |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['seed']} | {r['elapsed_seconds']:.1f} | {r['peak_cuda_allocated_bytes']/1024**2:.1f} | {r['peak_cuda_reserved_bytes']/1024**2:.1f} | {r['before']['metrics']['student']['kl_to_teacher']:.3f} | {r['after']['metrics']['student']['kl_to_teacher']:.3f} |")
    tokens = first["after"]["mean_tokens"]
    lines += ["", "Allocated/reserved figures are PyTorch measurements, not total driver or system VRAM use. "
              "The full backbone fit comfortably on the tested 16 GB card.", "",
              f"The four soft vectors occupy {first['soft_memory_bytes_per_example']:,} bytes per example. "
              f"For seed {first['seed']}, mean teacher input was {tokens['full']:.1f} tokens; student input was "
              f"{tokens['local']:.1f} recent/query tokens plus four soft positions. "
              f"The compressor also required a separate frozen-model pass over {tokens['old']:.1f} old-prefix tokens. "
              "This extra pass and compressor cost must be included in any efficiency comparison. No net speedup or end-to-end memory-saving claim is made.", "",
              "## Interpretation and next decision", "",
              "The learned vectors carry some answer-relevant information: intact memory outperforms absent or shuffled memory on the short task, "
              "and the predictive distribution moves toward the teacher. The remaining gap to full text is large. "
              "This is an early positive signal about the interface, not a practical replacement for full context.", "",
              "The current controls do not establish that the compressor reliably preserves object/color bindings rather than merely which colors appeared. "
              "A binding-swap control, a bag-of-colors baseline, new sentence templates, recent-query tests and repeated updates should precede bigger models or longer runs. "
              "Exact evidence retrieval is also an essential baseline before claiming practical value.", "",
              "Further limitations: small fixed vocabulary, one answer token, few training seeds, no matched-cost compression baseline, "
              "no statistical significance claim, and no total stream-memory measurement. Teacher selection used a small exploratory prompt probe; "
              "the training gate then used a different, larger set of 256 examples. The gate was 90% raw-token accuracy for this narrow task.", "",
              "## Reproduce", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-lm.txt"]
    for r in results:
        lines.append(f".\\.venv\\Scripts\\python.exe -m breadcrumb_memory.lm_memory --output runs/reproduce-lm-{r['seed']} --seed {r['seed']} --steps 1000 --max-minutes 15")
    lines += ["```", "", "Per-run JSON includes pinned model revision, source hashes, metrics and synthetic examples. "
              "Only those reviewed JSON reports are committed; caches and trained adapters remain local.", ""]
    Path("docs/LM_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("docs/LM_RESULTS.md")


if __name__ == "__main__":
    main()
