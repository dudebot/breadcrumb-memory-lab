# Breadcrumb Memory Lab

A small local research sandbox for this question:

> Can a fixed-size learned memory preserve useful information after it leaves a model's recent context, trained by comparison with a full-history teacher?

**Status: runnable synthetic ring pilot plus a frozen 135M language-model memory experiment, tested on an RTX 5070 Ti 16 GB.** The original integer-pair pilot needs no model download. The optional language-model experiment downloads SmolLM2-135M-Instruct and trains a small context compressor. Neither experiment implements a complete agent memory system, reproduces TTCD, or establishes a novel architecture.

The initial request was for a low-maintenance experiment that can run on a personal GPU. The research direction is interesting enough to measure; this repository deliberately makes the first measurement inexpensive. See [the research plan](docs/PLAN.md) for what remains before a language-model claim.

Start with [the plain-language explanation](docs/EXPLAINER.md) if “40 records” or “recall accuracy” is unclear. [Language-model results](docs/LM_RESULTS.md) describe the newer experiment separately from the original toy task.

## Quick start on Windows

Requires Python 3.10–3.13 for the pinned setup below, Git, and an NVIDIA driver compatible with the selected PyTorch wheel. Python 3.13, PyTorch 2.7.1+cu128, driver 576.88 and the 5070 Ti were tested locally.

```powershell
git clone https://github.com/dudebot/breadcrumb-memory-lab.git
cd breadcrumb-memory-lab
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Preset smoke
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Preset pilot
```

The setup script creates a local `.venv` and installs the CUDA 12.8 build of PyTorch. That is the only large download. If Python already has working CUDA PyTorch, use `scripts/setup.ps1 -UseExistingTorch` **when first creating the environment**. That option reuses system packages read-only and installs nothing globally. The initial local run used this option. The fresh-download installation recipe has not been exercised on a clean machine.

The scripts print a report and save artifacts in a timestamped `runs/` directory. A smoke run checks plumbing; its accuracy estimates are not meaningful. The pilot runs 100 teacher steps and 600 student steps with three basic model controls and exact lookup controls. Its configured time cap is ten minutes; on the tested GPU this tiny pilot takes seconds after import. CPU operation is supported with `-Device cpu`.

To use commands directly:

```powershell
.\.venv\Scripts\python.exe -m breadcrumb_memory doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m breadcrumb_memory train --config configs/pilot.json --output runs/my-pilot --device cuda
```

Run directories must be empty or new; existing results are never overwritten. There is no background service, automatic sweep, paid API, dataset upload, or unattended follow-up run.

## Try it manually

After the direct pilot command above:

```powershell
.\.venv\Scripts\python.exe -m breadcrumb_memory demo --checkpoint runs/my-pilot/checkpoint.pt --history examples/history.json
.\.venv\Scripts\python.exe -m breadcrumb_memory demo --checkpoint runs/my-pilot/checkpoint.pt --length 40 --query-position 0
.\.venv\Scripts\python.exe -m breadcrumb_memory demo --checkpoint runs/my-pilot/checkpoint.pt --length 96 --query-position 0
```

Edit the example's `records` and `query` to try another lookup. Keys are unique integers 0–127 and values are integers 0–31. Each query asks for the value associated with one key; it is not a chat prompt. The demo reports the right answer, predictions, recent records and whether the supporting record survived eviction. In the last command the oldest record has already been discarded; a wrong answer is expected.

For timestamped runs created by `scripts/run.ps1`, substitute the printed run path for `runs/my-pilot`.

## What is actually implemented?

This section describes the original integer-pair ring. The separate language-model experiment is described below.

Every episode contains newly randomized key/value bindings. The first model is a tiny trained content-attention teacher that can inspect the entire history. It has a learned key embedding, cosine attention and a fixed one-hot value encoding. It is intentionally easy for that teacher to solve this structured task. It is not a transformer language model.

The student has a separate learned key feature map. It retains eight recent raw records. Evicted records are written into a cyclic array of eight memory slots, with four records per slot:

1. Encode a record's key into a normalized, learned 16-dimensional vector.
2. Add its outer product with the one-hot value into the current slot.
3. Clear and reuse the oldest slot when the ring wraps.
4. At query time, read the accumulated associative memory with the query's feature vector, and combine it with the recent records.

Training minimizes `KL(teacher probabilities || student probabilities)` over 32 possible values. This has the same student gradient as soft-target cross-entropy. Gradients train the key feature map and a score scale; the teacher is frozen during student training. They do not prove identical hidden states.

**The FIFO schedule, value encoding and ring structure are hand-designed.** The feature map is learned; the write/forget policy is not. This first stage uses a single slow ring and has no learned archive pointer, retrieval controller, hierarchical timescales, text compressor, or tool integration. It is a baseline for the proposed mechanism, not the complete proposal.

The model writer has no query input. All records are written chronologically before answering. Training and evaluation share a key vocabulary but use independent random bindings. Evaluation examples are fixed across before/after checks so the comparison uses identical questions. The same evaluation set should not be used to tune a long series of variants; reserve another seed set for final evaluation if development continues.

## How to read the results

Each run saves `config.json`, `environment.json`, `metrics.jsonl`, `checkpoint.pt`, `results.json` and `REPORT.md`. Checkpoints contain model weights and configuration, not optimizer/RNG state: they support the demo, **not exact training resume**. Ctrl+C and the time cap save an interrupted checkpoint without claiming a completed evaluation. The cap is checked between operations, not a hard GPU-process kill deadline.

Important comparisons:

| Control | What it tells us |
|---|---|
| Student before versus after training | Whether learning improves on random associative features |
| Disabled memory | Whether recent context alone explains the answer |
| Other episodes' memory | Whether the correct historical content matters |
| Full-history teacher | Whether the distillation target can solve the task |
| Exact archive lookup | Whether a simpler symbolic lookup already solves the problem |
| Exact lookup within retained records | Whether the answer was lost to eviction or to representation interference |

Reports separate older and recent queries, and record accuracy conditional on evidence still being retained. JSON includes teacher/student KL, negative log-likelihood and binomial confidence intervals. Intervals describe episode sampling, not uncertainty across training seeds.

**Initial finding:** memory helps old-fact recall, training improves on an already strong random-feature baseline, and exact lookup remains best. Memory also interferes with recent-fact recall. See [the measured initial results](docs/INITIAL_RESULTS.md); these limitations are reasons to keep the next experiment small.

## Does 16 GB suffice?

Yes for this pilot, by a wide margin. `doctor` measures the actual device and launches a forward/backward CUDA operation. The initial machine had a 5070 Ti with 16,303 MiB reported VRAM and roughly 8 GB free. No driver or global ML package changes were made.

The default slow state is 8 × 16 × 32 × 4 = **16,384 bytes per episode**. That figure excludes weights, recent raw IDs, temporary tensors and the training graph. Autograd retains intermediate computations during training, so constant inference state does not imply constant training memory. The evaluation driver still holds its full synthetic episode to run controls.

This representation is **larger than storing this toy history as integer pairs**. It does not demonstrate real-world memory compression or VRAM savings over an LLM KV cache. It demonstrates a bounded, differentiable memory mechanism and a way to measure its failures.

A 7–8B BF16 language model's weights alone occupy roughly 14–16 GB before activations or training state. Quantized inference and some adapter-training setups can fit in 16 GB, but they add complexity. We instead tested a frozen 135M backbone with a small trainable compressor; see the measured memory use in the language-model report.

## Run the small language-model experiment

Optional dependencies and model downloads remain inside the local environment/cache. No account token is required. The default model revision is pinned, and remote model code is disabled.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lm.txt
.\.venv\Scripts\python.exe -m breadcrumb_memory.lm_probe --output runs/lm-probe.json
.\.venv\Scripts\python.exe -m breadcrumb_memory.lm_memory --output runs/lm-memory-17 --seed 17 --steps 1000 --max-minutes 15
```

This path requires CUDA. The default 135M model needs approximately 540 MB for FP32 weights, plus activations and temporaries. The runner caps its PyTorch allocator at at most 6 GiB (and at most 80% of device capacity), checks a time budget between operations, and saves an adapter if interrupted. It does not change the host driver, run in the background, or upload training data.

The task is to recover a color from four short factual sentences. The frozen model sees all four sentences as teacher. A 216,128-parameter compressor turns the first two sentences into four learned 576-dimensional vectors, then the same frozen model predicts from those vectors, the two recent sentences and the question. The writer never sees the question. Training uses the teacher's full-vocabulary next-token distribution, not a supervised answer loss.

Before training, the full-context teacher must achieve at least 90% correct raw next tokens on 256 held-out examples. This gate is specific to this deliberately short bridge experiment; it does not supersede the original synthetic teacher's 95% gate. Reports include eight-fact extrapolation, where this model is a weaker teacher.

**This is a single-prefix soft-memory experiment, not the cyclic ring from the integer task.** Recurrent writes, eviction, selective archive refresh and efficiency comparisons remain future work. The old text is encoded in a separate frozen-model pass; fewer student context positions do not imply lower total compute or constant-memory stream processing. Small evaluation sets, one answer token and a fixed sentence template also limit what the scores establish.

For a different model, supply both `--model` and its matching `--revision`; the defaults pin SmolLM2. Model weights and trained adapters stay out of Git.

## Relationship to prior work

[TTCD (August 2026)](https://arxiv.org/abs/2608.01672) uses discrepancies between longer- and shorter-context computations to supervise fast-weight memory. We share that supervision motivation. Our tiny pilot writes into an explicit associative ring rather than performing TTCD's MLP updates, and does not reproduce its language-model results. The authors' [implementation](https://github.com/dangxingyu/ttcd) is a useful separate reference.

[ICAE](https://arxiv.org/abs/2307.06945) studies learned compressed memory slots for language models. [Infini-attention](https://arxiv.org/abs/2404.07143) combines local attention with compressive memory, and [MemGPT](https://arxiv.org/abs/2310.08560) explores memory management across tiers. These make a broad novelty claim inappropriate. The outer-product memory here is deliberately a familiar linear associative mechanism.

The future question is whether **learned, query-independent breadcrumbs plus selective refresh** can beat simpler retrieval or compression under matched memory and total runtime budgets. The present experiment does not answer that yet.

## Repository map

- `breadcrumb_memory/`: generation, teacher, ring, training/evaluation, manual demo.
- `configs/`: short smoke and bounded pilot settings.
- `scripts/`: local setup, runs and result collection.
- `tests/`: causal writes, overwrite boundaries, constant state size, gradient flow and evaluation controls.
- `examples/`: manually editable history.
- `docs/PLAN.md`: staged research plan and stop conditions.
- `docs/EXPLAINER.md`: concrete examples of records, recall and the language-model follow-up.
- `docs/LM_RESULTS.md`: measured small-language-model results and limitations.
- `docs/INITIAL_RESULTS.md` and `results/initial/`: reviewed pilot results, not model checkpoints.
- `runs/`: ignored local logs and checkpoints.

No redistribution license has been selected for this repository. No external research code or model weights are vendored. The referenced SmolLM2 model has its own Apache-2.0 license; downloading it does not change this repository's licensing.
