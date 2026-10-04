"""Frozen small-LM experiment with query-independent learned soft memory.

Bridge experiment: compress one old text prefix into four input embeddings.
It does not yet implement streaming rings, archive retrieval or a complete agent.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .lm_probe import COLORS, color_ids, episode, load_model, render


class SoftMemory(nn.Module):
    def __init__(self, hidden_size, slots=4, bottleneck=128):
        super().__init__()
        self.slots = slots
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, bottleneck)
        self.queries = nn.Parameter(torch.randn(slots, bottleneck)*0.02)
        self.attention = nn.MultiheadAttention(bottleneck, 4, batch_first=True)
        self.output_norm = nn.LayerNorm(bottleneck)
        self.output_projection = nn.Linear(bottleneck, hidden_size)
        nn.init.normal_(self.output_projection.weight, std=0.002)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, old_hidden, old_mask):
        # Neither question nor answer is an argument to the writer.
        features = self.input_projection(self.input_norm(old_hidden.detach()))
        queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        read, _ = self.attention(queries, features, features,
                                 key_padding_mask=~old_mask.bool(), need_weights=False)
        return self.output_projection(self.output_norm(read+queries))


def batch(tokenizer, rng, size, records=4):
    items = [episode(rng, records) for _ in range(size)]
    old_text, local_text, full_text, answers = [], [], [], []
    for facts, target in items:
        query, answer = facts[target]
        # Only facts which will be removed: no query text or target is encoded here.
        old_text.append("Facts:\n"+"".join(f"The color of the {name} is {color}.\n" for name, color in facts[:-2]))
        local_text.append(render(facts[-2:], query))
        full_text.append(render(facts, query))
        answers.append(COLORS.index(answer))
    encode = lambda text: tokenizer(text, padding=True, return_tensors="pt").to("cuda")
    return {"old": encode(old_text), "local": encode(local_text), "full": encode(full_text),
            "answers": torch.tensor(answers, device="cuda"), "items": items}


def hidden(model, encoded):
    mask = encoded["attention_mask"]
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    return model.model(**encoded, position_ids=positions, use_cache=False).last_hidden_state


def next_logits(model, encoded):
    return model.get_output_embeddings()(hidden(model, encoded)[:, -1])


def student_logits(model, memory, local):
    local_embeddings = model.get_input_embeddings()(local["input_ids"])
    embeddings = torch.cat((memory, local_embeddings), dim=1)
    mask = torch.cat((torch.ones(memory.shape[:2], device=memory.device, dtype=local["attention_mask"].dtype), local["attention_mask"]), dim=1)
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    output = model.model(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions, use_cache=False)
    return model.get_output_embeddings()(output.last_hidden_state[:, -1])


@torch.no_grad()
def evaluate(model, tokenizer, compressor, ids, seed, examples, batch_size, records, deadline):
    rng = random.Random(seed)
    counts = {name: {"candidate_correct": 0, "raw_correct": 0, "kl_sum": 0.0}
              for name in ("teacher", "local", "student", "zero_memory", "shuffled_memory")}
    done = 0
    token_sums = {"old": 0, "local": 0, "full": 0}
    samples = []
    while done < examples:
        if time.perf_counter() >= deadline:
            raise TimeoutError("Time cap reached during LM evaluation")
        size = min(batch_size, examples-done)
        if size == 1:
            size = 2
        take = min(size, examples-done)
        data = batch(tokenizer, rng, size, records)
        features = hidden(model, data["old"])
        memory = compressor(features, data["old"]["attention_mask"])
        teacher = next_logits(model, data["full"])
        logits = {"teacher": teacher, "local": next_logits(model, data["local"]),
                  "student": student_logits(model, memory, data["local"]),
                  "zero_memory": student_logits(model, torch.zeros_like(memory), data["local"]),
                  "shuffled_memory": student_logits(model, memory.roll(1, dims=0), data["local"])}
        target = data["answers"][:take]
        for name, scores in logits.items():
            scores = scores[:take]
            counts[name]["candidate_correct"] += int((scores[:, ids].argmax(-1) == target).sum())
            counts[name]["raw_correct"] += int((scores.argmax(-1) == ids[target]).sum())
            counts[name]["kl_sum"] += float(F.kl_div(scores.log_softmax(-1), teacher[:take].softmax(-1), reduction="sum"))
        for name in token_sums:
            token_sums[name] += int(data[name]["attention_mask"][:take].sum())
        if len(samples) < 4:
            facts, target_pos = data["items"][0]
            query, answer = facts[target_pos]
            samples.append({"facts": facts, "query": query, "correct": answer,
                            "predictions": {name: COLORS[int(scores[0, ids].argmax())] for name, scores in logits.items()},
                            "raw_next_tokens": {name: tokenizer.decode([int(scores[0].argmax())]) for name, scores in logits.items()}})
        done += take
    return {"examples": examples, "records": records,
            "metrics": {name: {"candidate_accuracy": x["candidate_correct"]/examples,
                                "raw_next_token_accuracy": x["raw_correct"]/examples,
                                "kl_to_teacher": x["kl_sum"]/examples} for name, x in counts.items()},
            "mean_tokens": {name: value/examples for name, value in token_sums.items()},
            "memory_slots": compressor.slots, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--revision", default="12fd25f77366fa6b3b4b768ec3050bf629380bac")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-minutes", type=float, default=15)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.eval_examples < 2 or args.max_minutes <= 0:
        parser.error("steps >=1, batch-size >=2, eval-examples >=2, max-minutes >0 required")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.8, 6*1024**3/total))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    deadline = start+args.max_minutes*60
    model, tokenizer, revision = load_model(args.model, args.revision)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    ids = color_ids(tokenizer)
    compressor = SoftMemory(model.config.hidden_size).cuda()
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=0.001, weight_decay=0.01)
    completed = 0
    metadata = {"model": args.model, "revision": revision, "seed": args.seed, "steps_requested": args.steps,
                "batch_size": args.batch_size, "eval_examples": args.eval_examples,
                "backbone_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in compressor.parameters()),
                "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                "objective": "full-vocabulary forward KL; frozen backbone; four soft prefix embeddings",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name("lm_probe.py"))}}
    args.output.joinpath("config.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    def save(status):
        torch.save({"compressor": compressor.state_dict(), "metadata": metadata, "steps_completed": completed,
                    "status": status}, args.output/"adapter.pt")
    try:
        before = evaluate(model, tokenizer, compressor, ids, args.seed+90000, args.eval_examples, args.batch_size, 4, deadline)
        print(json.dumps({"phase": "before", "metrics": before["metrics"]}), flush=True)
        # A larger held-out gate prevents picking a prompt on 32 lucky examples.
        if before["metrics"]["teacher"]["raw_next_token_accuracy"] < 0.9:
            args.output.joinpath("teacher-gate-failed.json").write_text(json.dumps(before, indent=2)+"\n", encoding="utf-8")
            raise RuntimeError("Full-context teacher scored below 90%; stopped before training")
        for step in range(args.steps):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Time cap reached during LM training")
            data = batch(tokenizer, rng, args.batch_size)
            with torch.no_grad():
                old_hidden = hidden(model, data["old"])
                teacher = next_logits(model, data["full"]).softmax(-1)
            memory = compressor(old_hidden, data["old"]["attention_mask"])
            prediction = student_logits(model, memory, data["local"])
            loss = F.kl_div(prediction.log_softmax(-1), teacher, reduction="batchmean")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
            optimizer.step()
            completed = step+1
            if completed % 50 == 0 or completed == args.steps:
                record = {"phase": "train", "step": completed, "kl": float(loss.detach()), "elapsed_seconds": time.perf_counter()-start}
                with (args.output/"metrics.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record)+"\n")
                print(json.dumps(record), flush=True)
        after = evaluate(model, tokenizer, compressor, ids, args.seed+90000, args.eval_examples, args.batch_size, 4, deadline)
        longer = evaluate(model, tokenizer, compressor, ids, args.seed+190000, args.eval_examples, args.batch_size, 8, deadline)
        frozen = all(p.grad is None and not p.requires_grad for p in model.parameters())
        if not frozen:
            raise RuntimeError("Frozen-backbone invariant failed")
        torch.cuda.synchronize()
        result = {**metadata, "steps_completed": completed, "frozen_backbone_verified": frozen,
                  "elapsed_seconds": time.perf_counter()-start,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "soft_memory_bytes_per_example": 4*model.config.hidden_size*4,
                  "before": before, "after": after, "longer_history": longer}
        save("complete")
        args.output.joinpath("results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"phase": "complete", "metrics": after["metrics"], "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    except (TimeoutError, KeyboardInterrupt, RuntimeError) as exc:
        save("incomplete")
        args.output.joinpath("STOPPED.txt").write_text(f"{type(exc).__name__}: {exc}\nSteps completed: {completed}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
