# Initial measured results

## Scope

Pilot across 3 training seeds: 17, 29, 43. Each run used 100 teacher steps, 600 student steps, and 512 held-out episodes per length/query condition.

Device: **NVIDIA GeForce RTX 5070 Ti**. Python 3.13.5; PyTorch 2.7.1+cu128; CUDA runtime 12.8.
Teacher parameters: 8,193; student parameters: 2,049. This is a task-specific neural attention model, not a language model.

Measured run time: 10.2–10.5 seconds after process import, including evaluation. Peak PyTorch allocated GPU tensors: 21.4–21.4 MiB. This excludes the CUDA context, allocator reservation and other applications; it is not total process VRAM.

## Mean accuracy across training seeds

| Query | Records | Teacher | Before training | After training | Disabled memory | Shuffled memory | Exact archive | Evidence retained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old | 24 | 100.0% | 93.6% | 96.6% | 2.9% | 2.7% | 100.0% | 100.0% |
| old | 40 | 100.0% | 87.4% | 92.3% | 3.1% | 3.0% | 100.0% | 100.0% |
| old | 64 | 100.0% | 50.8% | 53.8% | 3.1% | 2.5% | 100.0% | 57.4% |
| old | 96 | 100.0% | 33.1% | 34.5% | 3.7% | 3.5% | 100.0% | 36.3% |
| recent | 24 | 100.0% | 94.4% | 96.8% | 99.7% | 90.3% | 100.0% | 100.0% |
| recent | 40 | 100.0% | 87.4% | 91.3% | 99.9% | 81.6% | 100.0% | 100.0% |
| recent | 64 | 100.0% | 85.1% | 90.8% | 99.8% | 82.3% | 100.0% | 100.0% |
| recent | 96 | 100.0% | 87.7% | 91.3% | 99.7% | 81.7% | 100.0% | 100.0% |

## Per-seed old-query results at 40 records

| Seed | Before | After | Improvement | Student KL |
|---:|---:|---:|---:|---:|
| 17 | 85.5% | 90.8% | +5.27 pp | 0.3767 |
| 29 | 88.9% | 91.8% | +2.93 pp | 0.2451 |
| 43 | 87.9% | 94.1% | +6.25 pp | 0.2046 |

## What this supports

The implemented memory carries information about old bindings, and training improves its feature geometry over random initialization. A control that removes or swaps the relevant memory loses that ability. This is a functional test of the prototype and evaluation, not a new result about frontier LLMs.

## What failed or remains unproved

- Random features already solve much of the task. Comparing only against memory-disabled performance would exaggerate the contribution of training.
- Exact keyed lookup reaches 100%. Nothing here establishes superiority over ordinary retrieval.
- Old memory interferes with recent queries; the memory-disabled reader is stronger on those. A gated local path is a candidate next test.
- Beyond 40 records, old supporting evidence is progressively overwritten. The model does not recover evidence that was truly lost.
- The 16,384-byte slow state is larger than the raw toy records. No real KV-cache compression ratio or end-to-end memory savings have been demonstrated.
- No pretrained LLM, learned retrieval controller, second timescale, conflicting updates or real tool logs have been tested.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m breadcrumb_memory train --config configs/pilot.json --output runs/reproduce-17 --device cuda --seed 17
.\.venv\Scripts\python.exe -m breadcrumb_memory train --config configs/pilot.json --output runs/reproduce-29 --device cuda --seed 29
.\.venv\Scripts\python.exe -m breadcrumb_memory train --config configs/pilot.json --output runs/reproduce-43 --device cuda --seed 43
```

The committed JSON files contain complete configurations, conditional accuracies, KL, sampling intervals and environment metadata. Each run records source hashes at execution time. `source-sha256.json` records source contents at collection time; per-run git revisions are recorded when a commit existed. No pretrained weights, local checkpoints, authentication material or conversation transcript are included.
