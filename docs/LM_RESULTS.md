# Small language-model experiment

## What was run

Backbone: `HuggingFaceTB/SmolLM2-135M-Instruct`, pinned to `12fd25f77366fa6b3b4b768ec3050bf629380bac`. All 134,515,008 backbone parameters were frozen. Only a 216,128-parameter compressor was trained.

Four factual sentences assign random colors to objects. The teacher sees all four. The student sees the last two plus four learned vectors encoding the first two. Those vectors are created before the query is supplied to the reader. The task is to predict the next color token in a fixed sentence-completion format.

Training uses full-vocabulary teacher/student KL, with no extra ground-truth answer loss. This is a single-prefix compression test. It has no recurrent ring, learned eviction or archive retrieval.

Each run completed 1,000 updates at batch size 8: 8,000 freshly randomized training episodes. Evaluation uses 256 independently generated examples per condition, the same before and after training. Training and evaluation share objects, colors and sentence templates; this does not test unseen language formats.

## Four-fact accuracy: actual highest-probability next token

| Seed | Full text | Recent text only | Untrained memory | Trained memory | Zero vectors | Other episode's memory |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 91.8% | 11.7% | 10.9% | 23.4% | 10.5% | 12.5% |
| 29 | 96.1% | 13.3% | 12.5% | 32.4% | 11.3% | 9.8% |

JSON additionally reports forced choice among eight color tokens. The table above uses the whole vocabulary. A uniform guess among the eight colors would be correct 12.5% of the time; raw next-token output is not constrained to those colors.

## Eight-fact extrapolation

| Seed | Full text | Recent only | Trained memory | Shuffled memory |
|---:|---:|---:|---:|---:|
| 17 | 63.3% | 10.2% | 14.5% | 12.5% |
| 29 | 60.9% | 10.5% | 19.1% | 11.3% |

The teacher itself is considerably weaker with eight facts. The long-history scores cannot establish reliable long-context capability.

## Resource measurements

GPU: NVIDIA GeForce RTX 5070 Ti; PyTorch 2.7.1+cu128. All model computation used FP32. Times include model loading from the local cache, training and all reported evaluation controls, but exclude the initial model download.

| Seed | Seconds | Peak allocated MiB | Peak reserved MiB | Teacher/student KL before | KL after |
|---:|---:|---:|---:|---:|---:|
| 17 | 155.2 | 833.9 | 844.0 | 1.419 | 0.803 |
| 29 | 164.5 | 834.7 | 844.0 | 1.542 | 0.843 |

Allocated/reserved figures are PyTorch measurements, not total driver or system VRAM use. The full backbone fit comfortably on the tested 16 GB card.

The four soft vectors occupy 9,216 bytes per example. For seed 17, mean teacher input was 50.2 tokens; student input was 32.1 recent/query tokens plus four soft positions. The compressor also required a separate frozen-model pass over 21.1 old-prefix tokens. This extra pass and compressor cost must be included in any efficiency comparison. No net speedup or end-to-end memory-saving claim is made.

## Interpretation and next decision

The learned vectors carry some answer-relevant information: intact memory outperforms absent or shuffled memory on the short task, and the predictive distribution moves toward the teacher. The remaining gap to full text is large. This is an early positive signal about the interface, not a practical replacement for full context.

The current controls do not establish that the compressor reliably preserves object/color bindings rather than merely which colors appeared. A binding-swap control, a bag-of-colors baseline, new sentence templates, recent-query tests and repeated updates should precede bigger models or longer runs. Exact evidence retrieval is also an essential baseline before claiming practical value.

Further limitations: small fixed vocabulary, one answer token, few training seeds, no matched-cost compression baseline, no statistical significance claim, and no total stream-memory measurement. Teacher selection used a small exploratory prompt probe; the training gate then used a different, larger set of 256 examples. The gate was 90% raw-token accuracy for this narrow task.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lm.txt
.\.venv\Scripts\python.exe -m breadcrumb_memory.lm_memory --output runs/reproduce-lm-17 --seed 17 --steps 1000 --max-minutes 15
.\.venv\Scripts\python.exe -m breadcrumb_memory.lm_memory --output runs/reproduce-lm-29 --seed 29 --steps 1000 --max-minutes 15
```

Per-run JSON includes pinned model revision, source hashes, metrics and synthetic examples. Only those reviewed JSON reports are committed; caches and trained adapters remain local.
