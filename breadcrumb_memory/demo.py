import json

import torch

from .data import Episode, generate, retained_start
from .experiment import Config, select_device
from .models import AssociativeRing, FullHistoryTeacher


@torch.no_grad()
def demo(checkpoint, history, length, seed, query_position, requested_device):
    device = select_device(requested_device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(**saved["config"])
    teacher = FullHistoryTeacher(cfg.n_keys, cfg.n_values, cfg.teacher_dim).to(device)
    student = AssociativeRing(cfg.n_keys, cfg.n_values, cfg.key_dim, cfg.slots, cfg.chunk_size, cfg.fast_window).to(device)
    teacher.load_state_dict(saved["teacher"])
    student.load_state_dict(saved["student"])
    teacher.eval()
    student.eval()
    if history:
        data = json.loads(history.read_text(encoding="utf-8"))
        records, query = data["records"], data["query"]
        if not records or len(records) > cfg.n_keys:
            raise ValueError("History must contain 1..n_keys records")
        keys = [r["key"] for r in records]
        values = [r["value"] for r in records]
        if any(type(k) is not int or not 0 <= k < cfg.n_keys for k in keys):
            raise ValueError(f"Keys must be integers in 0..{cfg.n_keys-1}")
        if any(type(v) is not int or not 0 <= v < cfg.n_values for v in values):
            raise ValueError(f"Values must be integers in 0..{cfg.n_values-1}")
        if len(set(keys)) != len(keys):
            raise ValueError("This pilot only supports unique keys; conflicting updates are a later stage")
        if type(query) is not int or query not in keys:
            raise ValueError("Query must be an integer key present in the history")
        pos = keys.index(query)
        e = Episode(torch.tensor([keys]), torch.tensor([values]), torch.tensor([query]),
                    torch.tensor([values[pos]]), torch.tensor([pos]))
    else:
        if not 0 <= query_position < length:
            raise ValueError("query-position must be within the history")
        e = generate(1, length, cfg.n_keys, cfg.n_values, cfg.fast_window, torch.Generator().manual_seed(seed))
        e.support[:] = query_position
        e.query = e.keys[:, query_position]
        e.target = e.values[:, query_position]
    e = e.to(device)
    predictions = {}
    for name, logits in (
        ("full_history_teacher", teacher(e.keys, e.values, e.query)),
        ("learned_ring", student(e.keys, e.values, e.query)),
        ("memory_disabled", student(e.keys, e.values, e.query, "no_memory")),
    ):
        probs = logits.softmax(-1)[0]
        top = probs.topk(min(3, cfg.n_values))
        predictions[name] = [{"value": int(v), "probability": round(float(p), 4)}
                             for p, v in zip(top.values, top.indices)]
    length = e.keys.shape[1]
    result = {"task": "synthetic key/value recall; probabilities are not calibrated confidence",
              "query_key": int(e.query.item()), "correct_value": int(e.target.item()),
              "support_position": int(e.support.item()), "history_length": length,
              "support_still_retained": bool(e.support.item() >= retained_start(length, cfg.fast_window, cfg.chunk_size, cfg.slots)),
              "recent_records": [{"key": k, "value": v} for k, v in zip(e.keys[0, -cfg.fast_window:].tolist(), e.values[0, -cfg.fast_window:].tolist())],
              "predictions": predictions}
    print(json.dumps(result, indent=2))
    return result
