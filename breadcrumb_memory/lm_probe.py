"""Check a real language model's full-context recall before memory training."""
import argparse
import json
import os
import random
import time
from pathlib import Path

# Keep all downloads local; no account token is needed for these public models.
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

OBJECTS = ("lantern", "bicycle", "notebook", "umbrella", "bottle", "chair", "table", "box",
           "pencil", "cup", "plate", "hat", "shirt", "coat", "bag", "book",
           "door", "clock", "ball", "car", "boat", "kite", "shoe", "lamp",
           "vase", "ribbon", "bucket", "fence", "bench", "helmet", "scarf", "basket")
COLORS = ("red", "blue", "green", "yellow", "black", "white", "orange", "purple")


def episode(rng, records):
    names = rng.sample(OBJECTS, records)
    colors = [rng.choice(COLORS) for _ in names]
    target = rng.randrange(max(1, records-2))
    return list(zip(names, colors)), target


def render(records, query, style="sentences"):
    if style == "sentences":
        return "Facts:\n" + "".join(f"The color of the {name} is {color}.\n" for name, color in records) + f"According to the facts, the color of the {query} is"
    if style == "mapping":
        return "Color assignments:\n" + "".join(f"{name}: {color}\n" for name, color in records) + f"\nRepeat the assigned color.\n{query}:"
    raise ValueError(style)


def load_model(model_id, revision=None):
    revision = revision or HfApi().model_info(model_id, token=False).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=False, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, token=False,
            trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32,
            attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)
    model.config.use_cache = False
    return model, tokenizer, revision


def color_ids(tokenizer):
    ids = [tokenizer.encode(" "+color, add_special_tokens=False) for color in COLORS]
    if any(len(x) != 1 for x in ids):
        raise ValueError("This probe requires each space-prefixed color to be one token")
    return torch.tensor([x[0] for x in ids], device="cuda")


@torch.no_grad()
def probe(model, tokenizer, examples=32):
    ids = color_ids(tokenizer)
    results = []
    for style in ("sentences", "mapping"):
        for length in (4, 8, 16):
            rng = random.Random(2010+length)
            correct = raw_correct = local_correct = 0
            samples = []
            for i in range(examples):
                records, pos = episode(rng, length)
                query, answer = records[pos]
                full = tokenizer(render(records, query, style), return_tensors="pt").to("cuda")
                local = tokenizer(render(records[-2:], query, style), return_tensors="pt").to("cuda")
                logits = model(**full).logits[0, -1]
                short = model(**local).logits[0, -1]
                choice = int(logits[ids].argmax())
                target = COLORS.index(answer)
                correct += choice == target
                raw_correct += int(logits.argmax()) == int(ids[target])
                local_correct += int(short[ids].argmax()) == target
                if i < 3:
                    samples.append({"prompt": render(records, query, style), "answer": answer,
                                    "candidate_prediction": COLORS[choice],
                                    "raw_next_token": tokenizer.decode([int(logits.argmax())])})
            row = {"style": style, "records": length, "examples": examples,
                   "full_context_candidate_accuracy": correct/examples,
                   "full_context_raw_token_accuracy": raw_correct/examples,
                   "local_candidate_accuracy": local_correct/examples,
                   "samples": samples}
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.examples <= 0:
        parser.error("examples must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model, tokenizer, revision = load_model(args.model, args.revision)
    rows = probe(model, tokenizer, args.examples)
    result = {"model": args.model, "revision": revision, "parameter_count": sum(p.numel() for p in model.parameters()),
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "elapsed_seconds_including_load": time.perf_counter()-start,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
