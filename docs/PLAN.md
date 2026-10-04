# Research plan

## Objective and scope

Investigate learned memory on one personal GPU without creating an ongoing operations burden. The original idea combined a small active context, slower learned breadcrumbs and optional refresh from archived evidence. Preserve the distinction between two hypotheses:

1. **Internal compression:** a bounded latent state preserves information useful for future predictions without consulting an archive.
2. **Selective refresh:** a bounded latent state helps locate and recover evidence from an external archive when needed.

The current pilot is an elementary instance of the first. The second is a separate experiment and must include archive storage, search and read costs. Neither implies an unbounded lossless memory in a finite state.

## Stage 0 — establish correct measurement (implemented)

Use unique-key synthetic histories and a tiny content-attention teacher. Train a learned key map for an associative ring by output-distribution distillation. Keep independent train/evaluation bindings. Test older and recent queries, multiple lengths, teacher quality, random initialization, disabled/shuffled memory, exact archive lookup and evidence survival.

Completed locally: GPU forward/backward check, unit tests, smoke run, three pilot seeds and a manual-query interface. See INITIAL_RESULTS.md for the observed numbers and versions. This code intentionally needs no pretrained model.

Acceptance means the experiment is correctly wired, not that the research hypothesis is solved:

- Teacher accuracy at least 95%; exact full archive lookup 100% on valid episodes.
- Old-query answers materially depend on the correct memory.
- State size stays fixed through many wraps; writes cannot see future queries.
- Training effects are compared against untrained features, across at least three seeds.
- Honest failure when supporting evidence is discarded, and explicit measurement of interference on recent queries.

## Stage 1 — test whether a better memory policy is worth pursuing

The first measured problem is interference with recent context. Before integrating an LLM:

- Train on a declared mixture of old and recent queries; evaluate each separately.
- Add a learned gate between local retrieval and compressed memory. Include a gate trained with no historical information as a control.
- Compare one ring with two timescales at equal **bytes**, including metadata. Compare against a single non-resetting associative accumulator and exact recent-record storage at the same byte budget.
- Add duplicate keys, timestamped corrections, obsolete facts and irrelevant distractors. The teacher/oracle must implement declared latest-version semantics; do not silently reuse the current unique-key teacher.
- Separate stream processing from evaluation storage and measure actual persistent state, peak allocation and all read/write latency.

Pre-register sizes, seeds and a final held-out seed set before selecting variants. A suggested success criterion is at least a five-percentage-point old-query gain over the best matched-budget simple baseline with no more than a two-point loss on recent queries, consistently across three seeds. These are proposed decision thresholds, not statistical guarantees.

Time budget: a bounded local work session with a fixed experiment list, not a large sweep. If the simple keyed retrieval baseline is all the intended application needs, stop and use it.

## Stage 2 — frozen small language model (initial bridge implemented)

An initial `lm_memory` experiment now uses frozen SmolLM2-135M-Instruct plus a learned four-vector compressor, on four-fact color recall. It is a single-prefix test, not yet a recurrent writer or the full benchmark below. The backbone remains frozen, and training targets are full-vocabulary teacher distributions. See LM_RESULTS.md for measured outcomes and constraints. The proposed breadth, repeated updates and efficiency controls below remain outstanding.

Choose a 100–500M causal language model with accessible weights, documented licensing, standard PyTorch support and a demonstrated ability to solve the test prompts. Pin model revision and tokenizer. Confirm the full-context teacher solves the tasks **before** interpreting a compressed student's loss.

Start with teacher context 512–2,048 tokens, recent student context 128–256 tokens, batch size one and a very small memory adapter. Profile before scaling. Frozen base weights reduce optimizer storage but gradients through frozen layers still have an activation cost. Avoid custom kernels initially; do not assume an H100 recipe works on a desktop Blackwell GPU.

Possible implementation: a query-independent recurrent compressor processes expiring chunks into a fixed bank of latent slots; a small adapter exposes those slots to selected backbone layers. Keep the backbone frozen. Train on later answer spans with KL to the same backbone using full historical context. Full-history targets can be precomputed sequentially to avoid two resident models. Log-probability targets and compression must not depend on future queries at write time.

Benchmarks should include randomized names/values, delayed questions, multiple supporting spans, corrections, paraphrased queries, distractor density and out-of-training-range delays. Teacher forcing is only the first test; evaluate generated answers and repeated updates too. Keep held-out content and templates, not merely different random seed labels on identical strings.

Required baselines: full context, local window, raw recent-token storage at equal bytes, simple textual summary, ordinary retrieval, oracle evidence retrieval, untrained adapter, disabled memory and shuffled memory. Include at least one suitable published learned-compression baseline before making a research claim.

The first bridge run logged timing every 50 steps and completed 1,000 steps under a 15-minute cap after passing a 256-example teacher gate. Record steps/second, peak allocated/reserved memory, device, exact configuration and target quality for later variants. Aim below 6 GiB of PyTorch allocation on the initially observed machine; stop on OOM rather than evicting other GPU users or modifying their processes.

A negative result here could reflect an inadequate interface, teacher or training procedure. Report which control failed instead of concluding that all compressed memory is impossible.

## Stage 3 — breadcrumbs that trigger retrieval (not implemented)

Keep an immutable archive of historical evidence on disk. A learned breadcrumb writer must encode useful retrieval cues before the future query exists. A reader ranks candidate chunks or emits an address; a refresh module feeds retrieved evidence or a learned representation into the active context.

Start with supervised evidence addresses from generated data and oracle retrieval as an upper reference. Hard top-k/address selection is discrete: cross-entropy on downstream text does not automatically backpropagate through file I/O. Use supervised retrieval first; differentiable mixtures or policy-gradient methods are later alternatives, with their costs disclosed.

Train prediction fidelity with explicit penalties for bytes retained, retrieval calls and tokens/bytes reread. Compare to ordinary lexical/dense retrieval using the same archive and candidate budget. Account for any growing search index: a bounded ring does not make the external index constant-size. GPU memory, host memory, disk, transfer bandwidth and runtime must all be reported separately. On Spark, CPU/GPU share the same physical memory pool.

Test stale and changing external state. Archived output answers "what did the command say then?"; rerunning a command answers "what is true now?" These are different tasks. Use synthetic recorded tool events before any live bot or filesystem integration.

Potential distinction from TTCD: a learned bounded cue store with selective, grounded evidence refresh, evaluated on the tradeoff between fidelity and retrieval cost. This is a candidate contribution, not an established novelty claim. A broader literature check is necessary before publication.

## Compute and maintenance policy

- No 7–8B training, DGX requirement or remote GPU rental in the initial scope.
- Do not use larger GPU capacity as a substitute for a diagnostic baseline.
- Runs are explicit commands with step and time limits; no automatic training after the command ends.
- Preserve small result JSON/report files; keep environments and checkpoints out of Git.
- Keep dependency/runtime changes local; do not upgrade the host driver or global PyTorch as an experiment side effect.
- A new repo is enough. No bot dependency, background daemon, cloud service or long-running CI GPU runner is needed.

## When to hand it off or stop

Hand off a short description, code, source links and measured results if the next stage needs expertise the owner does not want to maintain. Stop if ordinary retrieval wins the desired task at equal cost, if improvements vanish across seeds, or if the teacher cannot solve the benchmark. Advance to larger models only after a smaller experiment shows a defensible tradeoff.

Success for this repository can simply be a reproducible negative result that saves further work.
