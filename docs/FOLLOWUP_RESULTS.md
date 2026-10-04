# Follow-up: relationship memory, two rings, and step refresh

The follow-up produced a useful negative result for the learned compressor and a
working systems demonstration of the double circular buffer. They remain separate
experiments in one repository. Neither needs a larger GPU.

## 1. The learned memory does not yet preserve reliable relationships

The previous accuracy gain could come from remembering which colors appeared,
without remembering which object had each color. We tested the two saved adapters
without further training using the [declared protocol](FOLLOWUP_PROTOCOL.md).

Each test unit asks four questions:

1. With `cup=red, book=blue`, ask the cup's color and the book's color.
2. Swap the assignments to `cup=blue, book=red` and ask both questions again.

The other facts and question wording stay fixed. The same memory is used for both
questions within each history. A deterministic memory that only knows "red and
blue occurred" cannot answer all four correctly. Neither can a reader that always
repeats the first old color regardless of the question.

There are **256 independently generated quartets**, giving 1,024 correlated
answers per condition. The independent sample count is 256, not 1,024. We retain
teacher failures in the primary scores. All numbers below use the actual
highest-probability next token over the entire vocabulary.

| Condition | Individual answers correct | All four correct |
|---|---:|---:|
| SmolLM2 full text | 95.0% | 212/256 (82.8%) |
| Recent text only | 12.3% | 0/256 |
| Trained adapter, seed 17 | 22.0% | 0/256 |
| Seed 17, assignments erased before encoding | 23.3% | 0/256 |
| Trained adapter, seed 29 | 30.1% | 0/256 |
| Seed 29, assignments erased before encoding | 30.1% | 0/256 |
| Deterministic symbolic color-bag baseline | 50.0% | 0/256 |
| Exact symbolic key/value lookup | 100.0% | 256/256 |

The assignment-erased control canonicalizes the two old colors, producing the
same memory for both swapped histories. It is an ablation, not an optimized
bag-of-colors model. The symbolic bag baseline always chooses one of the two
distinct colors and therefore gets exactly half the individual answers correct.

This substantially weakens the earlier positive interpretation. The vectors
transfer answer-relevant information, but these runs do **not demonstrate reliable
object/color binding**. Occasional correctly answered pairs and sensitivity to a
swap do not establish complete absence of binding either. The narrower conclusion
is that reliable binding has not passed its diagnostic test.

The full-text teacher solves most quartets, so enlarging the backbone is not the
first repair to make. The writer/reader interface and training signal need work.
This failure does not disprove compressed memory or the separate refresh idea.

## 2. GPT-2 is a feasible comparison, but a weaker teacher on this format

We ran the dense 124,439,808-parameter
[GPT-2 checkpoint](https://huggingface.co/openai-community/gpt2), pinned to
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, on the same quartet histories.
It reached **52.2% individual accuracy and 7/256 complete quartets (2.7%)**.
Recent text alone reached 14.7%. No GPT-2 compressor was trained: its full-text
performance would confound a direct memory comparison on this task.

An additional exploratory probe used 256 examples per format/length:

| Facts | Sentence format, full text | Key/value mapping format, full text |
|---:|---:|---:|
| 4 | 57.8% | 88.7% |
| 8 | 60.5% | 86.3% |
| 16 | 56.3% | 77.7% |

Prompt format clearly matters. Mapping-format results are exploratory and do not
replace the failed sentence-format comparison or constitute a passed independent
training gate. The larger probe followed an initial 32-example exploration;
no model weights were tuned on either set.

All GPU evaluations ran locally on the 16 GB RTX 5070 Ti. The SmolLM2 quartet
evaluation used about 569 MiB peak PyTorch allocation and took 29 seconds; the
GPT-2 format probe used about 567 MiB and took 22 seconds. These exclude initial
downloads and do not measure total driver VRAM or training requirements. Hardware
capacity is not the present bottleneck.

## 3. Two real circular buffers, with refresh between simulated steps

The CPU lab implements four active records, sixteen slower `(key, address)`
breadcrumbs, and an external archive. Both rings overwrite old entries. Refresh
can happen at a simulated step boundary using an already-visible task cue, or at
question time. A policy never receives the future question early.

The final run uses opaque randomized keys independent of archive addresses,
three seeds (17, 29, 43), 1,000 episodes per scenario per seed, six scenarios and
nine policies: **162,000 episode-policy evaluations**. Each policy receives the
same generated episodes within a seed/scenario. Results below pool the three
equal-sized seed sets.

| Scenario | Step refresh, dual ring | Question-time refresh, dual ring | Growing exact index |
|---|---:|---:|---:|
| Stable cue, short delay, identical actual reads | 100% | 100% | 100% |
| Stable cue, variable delay | 50.2% | 62.1% | 100% |
| No early cue | 0% | 63.7% | 100% |
| Task changes after initial prefetch | 0% | 56.5% | 100% |
| New observed version after initial prefetch | 52.4% | 76.5% | 100% |
| Breadcrumb expired before the cue | 0% | 0% | 100% |

In the stable, short-delay scenario, both dual policies perform exactly **one
archive read**. Step refresh does it before the question; demand refresh does it
afterward. That shows work moving off the answer path, not a measured speedup or
actual asynchronous execution. The single raw ring under the same byte ceiling
scores 48.8%; corrupted pointers score 0%.

The modeled active budget is **440 bytes**, including ring/controller metadata.
The single ring leaves 16 bytes unused under that ceiling. Archives grow, and the
exact-lookup baseline pays for a growing index. These are logical packed sizes,
not Python process memory. Equal read caps do not imply equal consumed work
outside the dedicated timing scenario.

An early fetch can be wasted when the task changes; the fetched record can also
be overwritten before use. Once both fact and breadcrumb expire, the proposed
mechanism cannot find that evidence even though it remains in the archive. This
is a concrete limit, not a reason to hide the archive/index costs.

The [design report](STEP_MEMORY_DESIGN.md) details the controls and accounting.
This lab uses explicit keys, scripted cues, and one final query per episode. It
does not yet test learned semantic routing, multi-query agent sessions, actual
tool execution, KV-cache savings, or proprietary model internals.

## Independent implementation review

A second AI agent audited the methodology and implementation; this is not formal
peer review. It identified two important weaknesses:

- A simple color-swap pair was insufficient: both objects must be queried before
  and after the swap to rule out a query-ignoring shortcut. The final quartet test
  does this.
- The first CPU generator used sequential keys equal to archive addresses,
  permitting a hypothetical direct-address shortcut. The final generator uses
  randomized opaque keys; a regression test checks this, and all final CPU
  results were rerun. Earlier sequential-key pilot numbers are superseded.

The review also checked causal inputs, ring overwrite behavior, budget accounting,
version handling and matched actual reads. Remaining limits are stated above.

## Next productive step

Keep these controls fixed. On the same 135M backbone, compare a structured writer
that retains separate key/value information with the current pooled soft prefix,
and a query-conditioned reader that selects among those already-written slots.
The writer must still be blind to future questions. Use paired training queries
so the reader cannot succeed merely by naming a color that occurred. Compare
teacher-distribution training with explicitly supervised retrieval as separately
labeled objectives; a supervised win would not establish a distillation-only win.

Declare fresh validation/final seeds before selecting a variant. Require complete
quartet performance materially above binding-blind controls across training seeds
before connecting it to recurrent rings or increasing the model size. Then test
multiple queries per trajectory, budget allocation across steps, real retrieval
latency, and ordinary retrieval baselines. The circular-buffer direction remains
part of the project; it is not contingent on the current compressor succeeding.

## Reproduce and inspect

Raw reviewed reports: [SmolLM2 bindings](../results/followup/binding-smollm.json),
[GPT-2 bindings](../results/followup/binding-gpt2.json),
[GPT-2 formats](../results/followup/gpt2-probe.json), and
[CPU seeds](../results/step-memory/). Model revisions, synthetic samples and neural
source hashes are included. A [collection manifest](../results/followup/manifest.json)
records hashes of the CPU implementation/tests and all collected reports.

First train the two adapters using the commands in [LM_RESULTS.md](LM_RESULTS.md).
They are kept out of Git. Then:

```powershell
.venv\Scripts\python.exe -m breadcrumb_memory.binding_eval --checkpoints runs/reproduce-lm-17/adapter.pt runs/reproduce-lm-29/adapter.pt --output runs/binding-smollm.json
.venv\Scripts\python.exe -m breadcrumb_memory.binding_eval --model openai-community/gpt2 --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --output runs/binding-gpt2.json
.venv\Scripts\python.exe -m breadcrumb_memory.lm_probe --model openai-community/gpt2 --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --examples 256 --output runs/gpt2-probe.json
.venv\Scripts\python.exe -m breadcrumb_memory.step_memory --output runs/step-17 --episodes 1000 --seed 17
.venv\Scripts\python.exe -m breadcrumb_memory.step_memory --output runs/step-29 --episodes 1000 --seed 29
.venv\Scripts\python.exe -m breadcrumb_memory.step_memory --output runs/step-43 --episodes 1000 --seed 43
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use new output paths. No paid APIs, unattended services, uploaded private data,
larger GPUs or model checkpoints in the repository are needed.
