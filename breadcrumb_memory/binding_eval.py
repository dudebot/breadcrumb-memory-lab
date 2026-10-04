"""Counterfactual quartets test object/value bindings, not merely color presence."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from .lm_memory import SoftMemory, hidden, next_logits, student_logits
from .lm_probe import COLORS, OBJECTS, color_ids, load_model, render


def quartets(rng, count):
    examples = []
    for _ in range(count):
        names = rng.sample(OBJECTS, 4)
        a, b = rng.sample(COLORS, 2)
        recent = [(names[2], rng.choice(COLORS)), (names[3], rng.choice(COLORS))]
        original = [(names[0], a), (names[1], b)] + recent
        swapped = [(names[0], b), (names[1], a)] + recent
        # Canonicalization erases the assignment but preserves names and multiset.
        low, high = sorted((a, b))
        blind = [(names[0], low), (names[1], high)] + recent
        examples.append({"original": original, "swapped": swapped, "blind": blind,
                         "queries": names[:2], "answers": [a, b, b, a]})
    return examples


def old_text(facts):
    return "Facts:\n"+"".join(f"The color of the {name} is {color}.\n" for name, color in facts[:2])


def texts(examples):
    old, blind, full, local, answers = [], [], [], [], []
    for example in examples:
        blind.append(old_text(example["blind"]))
        answers.append([COLORS.index(x) for x in example["answers"]])
        for name in ("original", "swapped"):
            facts = example[name]
            old.append(old_text(facts))
            for query in example["queries"]:
                full.append(render(facts, query))
                local.append(render(facts[-2:], query))
    return {"old": old, "blind": blind, "full": full, "local": local, "answers": answers}


def score(predictions, answers):
    """Each row: original/query0, original/query1, swapped/query0, swapped/query1."""
    right = predictions == answers
    return {"per_answer": float(right.float().mean()),
            "both_queries_within_history": float(right.reshape(-1, 2).all(-1).float().mean()),
            "all_four": float(right.all(-1).float().mean()),
            "both_assignments_for_same_query": float((right[:, :2] & right[:, 2:]).float().mean()),
            "prediction_changes_after_swap": float((predictions[:, :2] != predictions[:, 2:]).float().mean())}


@torch.no_grad()
def evaluate(model, tokenizer, compressor, examples, batch_size=4):
    ids = color_ids(tokenizer)
    predictions = {}
    candidate_predictions = {}
    targets = []
    for start in range(0, len(examples), batch_size):
        text = texts(examples[start:start+batch_size])
        encode = lambda values: tokenizer(values, padding=True, return_tensors="pt").to("cuda")
        full, local = encode(text["full"]), encode(text["local"])
        logits = {"teacher": next_logits(model, full), "local": next_logits(model, local)}
        if compressor is not None:
            old, blind = encode(text["old"]), encode(text["blind"])
            # Each history is encoded once. Repeat its memory for the two readers.
            memory = compressor(hidden(model, old), old["attention_mask"]).repeat_interleave(2, dim=0)
            blind_memory = compressor(hidden(model, blind), blind["attention_mask"]).repeat_interleave(4, dim=0)
            logits.update(student=student_logits(model, memory, local),
                          binding_blind=student_logits(model, blind_memory, local),
                          zero_memory=student_logits(model, torch.zeros_like(memory), local))
        answer = torch.tensor(text["answers"])
        targets.append(answer)
        for name, values in logits.items():
            predictions.setdefault(name, []).append(values.argmax(-1).reshape(-1, 4).cpu())
            candidate_predictions.setdefault(name, []).append(values[:, ids].argmax(-1).reshape(-1, 4).cpu())
    answer = torch.cat(targets)
    actual_tokens = ids.cpu()[answer]
    raw = {k: torch.cat(v) for k, v in predictions.items()}
    candidate = {k: torch.cat(v) for k, v in candidate_predictions.items()}
    teacher_good = (raw["teacher"] == actual_tokens).all(-1)
    result = {"quartets": len(examples), "answers": 4*len(examples),
              "raw_next_token": {k: score(v, actual_tokens) for k, v in raw.items()},
              "eight_color_choice": {k: score(v, answer) for k, v in candidate.items()},
              "teacher_all_four_correct_quartets": int(teacher_good.sum()),
              "raw_on_teacher_correct_quartets": {k: score(v[teacher_good], actual_tokens[teacher_good]) for k, v in raw.items()} if teacher_good.any() else None}
    # A deterministic multiset-only baseline always chooses the same color.
    bag = torch.tensor([[COLORS.index(sorted((e["answers"][0], e["answers"][1]))[0])]*4 for e in examples])
    result["symbolic_bag_baseline"] = score(bag, answer)
    lookup = torch.tensor([[COLORS.index(dict(e[history])[query])
                            for history in ("original", "swapped") for query in e["queries"]]
                           for e in examples])
    result["exact_symbolic_lookup"] = score(lookup, answer)
    result["samples"] = [{**e, "predicted_colors": {k: [COLORS[x] for x in v[i].tolist()] for k, v in candidate.items()}}
                         for i, e in enumerate(examples[:4])]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="*")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--quartets", type=int, default=256)
    parser.add_argument("--seed", type=int, default=34001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.quartets < 1:
        parser.error("quartets must be positive")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_per_process_memory_fraction(min(0.8, 6*1024**3/torch.cuda.get_device_properties(0).total_memory))
    start = time.perf_counter()
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in (args.checkpoints or [])]
    if saved:
        if any(s["status"] != "complete" for s in saved):
            raise ValueError("Only complete checkpoints are accepted")
        if any((s["metadata"]["model"], s["metadata"]["revision"]) != (saved[0]["metadata"]["model"], saved[0]["metadata"]["revision"]) for s in saved):
            raise ValueError("Checkpoints must share backbone and revision")
        model_id, revision = saved[0]["metadata"]["model"], saved[0]["metadata"]["revision"]
    else:
        model_id, revision = args.model, args.revision
    model, tokenizer, revision = load_model(model_id, revision)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    examples = quartets(random.Random(args.seed), args.quartets)
    runs = []
    for checkpoint in saved or [None]:
        compressor = None
        if checkpoint is not None:
            compressor = SoftMemory(model.config.hidden_size).cuda().eval()
            compressor.load_state_dict(checkpoint["compressor"])
        result = evaluate(model, tokenizer, compressor, examples)
        result["adapter_training_seed"] = checkpoint["metadata"]["seed"] if checkpoint else None
        runs.append(result)
        print(json.dumps({"adapter_seed": result["adapter_training_seed"], "raw_next_token": result["raw_next_token"]}), flush=True)
    output = {"model": model_id, "revision": revision, "parameters": sum(p.numel() for p in model.parameters()),
              "test_seed": args.seed, "quartets": args.quartets, "elapsed_seconds": time.perf_counter()-start,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name("lm_memory.py"), Path(__file__).with_name("lm_probe.py")]},
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
