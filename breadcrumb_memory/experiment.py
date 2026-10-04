import hashlib
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import exact_retrieval, generate, retained_start
from .models import AssociativeRing, FullHistoryTeacher


@dataclass
class Config:
    seed: int = 17
    n_keys: int = 128
    n_values: int = 32
    teacher_dim: int = 64
    key_dim: int = 16
    slots: int = 8
    chunk_size: int = 4
    fast_window: int = 8
    batch_size: int = 64
    teacher_steps: int = 100
    student_steps: int = 600
    learning_rate: float = 0.01
    train_lengths: tuple = (16, 24, 32, 40)
    eval_lengths: tuple = (24, 40, 64, 96)
    eval_examples: int = 512
    max_minutes: float = 10

    def validate(self):
        for field in ("n_keys", "n_values", "teacher_dim", "key_dim", "slots",
                      "chunk_size", "fast_window", "batch_size", "eval_examples"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >=2 for shuffled-memory control")
        if not self.train_lengths or not self.eval_lengths:
            raise ValueError("length lists must not be empty")
        if any(t <= self.fast_window or t > self.n_keys for t in (*self.train_lengths, *self.eval_lengths)):
            raise ValueError("lengths must exceed fast_window and be <= n_keys")
        if self.teacher_steps < 1 or self.student_steps < 0:
            raise ValueError("teacher_steps >=1 and student_steps >=0 are required")
        if self.max_minutes <= 0 or self.learning_rate <= 0:
            raise ValueError("max_minutes and learning_rate must be positive")


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use doctor or --device cpu")
    return torch.device(requested)


def hardware(test_cuda=True):
    result = {"python": platform.python_version(), "platform": platform.system(),
              "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(gpu=torch.cuda.get_device_name(), free_vram_bytes=free,
                      total_vram_bytes=total, capability=list(torch.cuda.get_device_capability()))
        if test_cuda:
            # Actually launch kernels: driver visibility alone is insufficient.
            x = torch.randn(16, 16, device="cuda", requires_grad=True)
            (x @ x.T).square().mean().backward()
            torch.cuda.synchronize()
            result["cuda_forward_backward_passed"] = bool(torch.isfinite(x.grad).all().item())
    return result


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def wilson(successes, count):
    if not count:
        return None
    p, z = successes / count, 1.96
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    radius = z * math.sqrt(p * (1-p) / count + z*z / (4*count*count)) / denom
    return [max(0, center-radius), min(1, center+radius)]


@torch.no_grad()
def evaluate(teacher, student, cfg, device, deadline=None):
    teacher.eval()
    student.eval()
    rows = []
    for query_mode in ("old", "recent"):
        for length in cfg.eval_lengths:
            # Identical evaluation episodes before and after student training.
            rng = torch.Generator().manual_seed(cfg.seed + 100_000 + length + (10_000 if query_mode == "recent" else 0))
            totals = {name: 0 for name in ("teacher", "recent_teacher", "student", "no_memory", "shuffled", "exact_archive", "exact_retained")}
            count = retained = right_retained = right_lost = 0
            kl = nll = 0.0
            start = time.perf_counter()
            while count < cfg.eval_examples:
                if deadline is not None and time.perf_counter() > deadline:
                    raise TimeoutError("Time cap reached during evaluation; reduce eval_examples or increase max_minutes")
                size = min(cfg.batch_size, cfg.eval_examples - count)
                # Avoid a one-item shuffled batch being identical to itself.
                if size == 1:
                    size = 2
                episode = generate(size, length, cfg.n_keys, cfg.n_values, cfg.fast_window, rng, query_mode).to(device)
                target = episode.target
                teacher_logits = teacher(episode.keys, episode.values, episode.query)
                recent_logits = teacher(episode.keys[:, -cfg.fast_window:], episode.values[:, -cfg.fast_window:], episode.query)
                state, rk, rv = student.encode(episode.keys, episode.values)
                logits = student.read(state, rk, rv, episode.query)
                predictions = {"teacher": teacher_logits.argmax(-1), "recent_teacher": recent_logits.argmax(-1),
                               "student": logits.argmax(-1),
                               "no_memory": student.read(state, rk, rv, episode.query, "no_memory").argmax(-1),
                               "shuffled": student.read(state, rk, rv, episode.query, "shuffled").argmax(-1),
                               "exact_archive": exact_retrieval(episode),
                               "exact_retained": exact_retrieval(episode, retained_start(length, cfg.fast_window, cfg.chunk_size, cfg.slots))}
                take = min(size, cfg.eval_examples-count)
                for name, prediction in predictions.items():
                    totals[name] += int((prediction[:take] == target[:take]).sum().item())
                keep = episode.support[:take] >= retained_start(length, cfg.fast_window, cfg.chunk_size, cfg.slots)
                correct = predictions["student"][:take] == target[:take]
                retained += int(keep.sum().item())
                right_retained += int((keep & correct).sum().item())
                right_lost += int((~keep & correct).sum().item())
                log_student = logits[:take].log_softmax(-1)
                prob_teacher = teacher_logits[:take].softmax(-1)
                kl += float(F.kl_div(log_student, prob_teacher, reduction="sum").item())
                nll += float(F.cross_entropy(logits[:take], target[:take], reduction="sum").item())
                count += take
            sync(device)
            rows.append({"length": length, "query_mode": query_mode, "examples": count,
                         "accuracy": {k: v/count for k, v in totals.items()},
                         "student_wilson95": wilson(totals["student"], count),
                         "support_retained_fraction": retained/count,
                         "accuracy_when_retained": right_retained/retained if retained else None,
                         "accuracy_when_lost": right_lost/(count-retained) if retained < count else None,
                         "teacher_student_kl": kl/count, "student_nll": nll/count,
                         "evaluation_seconds_all_controls": time.perf_counter()-start})
    return rows


def report_markdown(result):
    cfg = result["config"]
    lines = ["# Synthetic pilot results", "", "This is a tiny associative-recall experiment, not an LLM benchmark or a TTCD reproduction.", "",
             f"Seed: {cfg['seed']}. Device: {result['device']}. Completed student steps: {result['completed_student_steps']}.",
             f"Persistent slow state: {result['slow_state_bytes_per_episode']:,} bytes per episode (FP32).",
             f"Elapsed: {result['elapsed_seconds']:.1f} s. Peak PyTorch allocated GPU memory: {result['peak_cuda_allocated_bytes']:,} bytes.", "",
             "Accuracy on freshly generated bindings; old queries refer to evidence outside the recent window.", "",
             "| Queries | Records | Teacher | Student before | Student after | No memory | Shuffled | Exact archive | Support retained |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for before, after in zip(result["before"], result["after"]):
        a = after["accuracy"]
        lines.append(f"| {after['query_mode']} | {after['length']} | {a['teacher']:.1%} | {before['accuracy']['student']:.1%} | {a['student']:.1%} | {a['no_memory']:.1%} | {a['shuffled']:.1%} | {a['exact_archive']:.1%} | {after['support_retained_fraction']:.1%} |")
    lines += ["", "## Interpretation boundaries", "",
              "- Randomly initialized features already provide associative recall; compare before/after, not just against chance.",
              "- The archive control is an exact dictionary lookup on structured keys. It should win this deliberately simple task.",
              "- FIFO eviction deliberately loses records. Accuracy beyond retention capacity is not evidence of unlimited memory.",
              "- Memory bytes exclude model weights, the recent buffer, autograd, and temporary tensors. They are not total VRAM savings.",
              "- This memory is larger than the raw integer records in this toy task. No compression-efficiency win is claimed.",
              "- Timing covers all controls; it is not a throughput comparison. See results.json for KL, conditional accuracy and confidence intervals.",
              "- Confidence intervals cover sampled episodes, not variability across training seeds.", ""]
    return "\n".join(lines)


def run(cfg, output, requested_device):
    cfg.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(requested_device)
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    deadline = start + cfg.max_minutes*60
    rng = torch.Generator().manual_seed(cfg.seed)
    length_rng = random.Random(cfg.seed)
    teacher = FullHistoryTeacher(cfg.n_keys, cfg.n_values, cfg.teacher_dim).to(device)
    student = AssociativeRing(cfg.n_keys, cfg.n_values, cfg.key_dim, cfg.slots, cfg.chunk_size, cfg.fast_window).to(device)
    output.joinpath("config.json").write_text(json.dumps(asdict(cfg), indent=2)+"\n", encoding="utf-8")
    environment = hardware(test_cuda=device.type == "cuda")
    output.joinpath("environment.json").write_text(json.dumps(environment, indent=2)+"\n", encoding="utf-8")
    log_path = output / "metrics.jsonl"

    def log(record):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)

    def checkpoint(completed, status):
        torch.save({"config": asdict(cfg), "teacher": teacher.state_dict(), "student": student.state_dict(),
                    "completed_student_steps": completed, "status": status}, output/"checkpoint.pt")

    completed = 0
    try:
        optimizer = torch.optim.Adam(teacher.parameters(), lr=cfg.learning_rate)
        for step in range(cfg.teacher_steps):
            if time.perf_counter() > deadline:
                raise TimeoutError("Time cap reached during teacher training")
            episode = generate(cfg.batch_size, length_rng.choice(cfg.train_lengths), cfg.n_keys, cfg.n_values, cfg.fast_window, rng).to(device)
            loss = F.nll_loss(teacher(episode.keys, episode.values, episode.query), episode.target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        teacher.eval().requires_grad_(False)
        log({"phase": "teacher", "steps": cfg.teacher_steps, "nll": loss.item()})
        before = evaluate(teacher, student, cfg, device, deadline)
        if min(row["accuracy"]["teacher"] for row in before) < 0.95:
            raise RuntimeError("Teacher accuracy below 95%; do not interpret student distillation")
        optimizer = torch.optim.Adam(student.parameters(), lr=cfg.learning_rate)
        student.train()
        for step in range(cfg.student_steps):
            if time.perf_counter() > deadline:
                raise TimeoutError("Time cap reached during student training")
            episode = generate(cfg.batch_size, length_rng.choice(cfg.train_lengths), cfg.n_keys, cfg.n_values, cfg.fast_window, rng).to(device)
            with torch.no_grad():
                target = teacher(episode.keys, episode.values, episode.query).softmax(-1)
            logits = student(episode.keys, episode.values, episode.query)
            loss = F.kl_div(logits.log_softmax(-1), target, reduction="batchmean")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stopped")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            completed = step+1
            if completed % 100 == 0 or completed == cfg.student_steps:
                log({"phase": "student", "step": completed, "kl": loss.item(), "elapsed_seconds": time.perf_counter()-start})
        after = evaluate(teacher, student, cfg, device, deadline)
        sync(device)
        result = {"schema_version": 1, "config": asdict(cfg), "device": str(device),
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "environment": environment, "completed_student_steps": completed,
                  "slow_state_bytes_per_episode": student.state_bytes(),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  "elapsed_seconds": time.perf_counter()-start,
                  "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
                  "student_parameters": sum(p.numel() for p in student.parameters()),
                  "before": before, "after": after}
        try:
            result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            result["git_commit"] = None
        checkpoint(completed, "complete")
        output.joinpath("results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        output.joinpath("REPORT.md").write_text(report_markdown(result), encoding="utf-8")
        log({"phase": "complete", "report": str(output/"REPORT.md"), "elapsed_seconds": result["elapsed_seconds"]})
        return result
    except (TimeoutError, KeyboardInterrupt) as exc:
        checkpoint(completed, "interrupted")
        output.joinpath("INTERRUPTED.txt").write_text(f"{type(exc).__name__}: {exc}\nCompleted student steps: {completed}\nNo final evaluation claim.\n", encoding="utf-8")
        raise
