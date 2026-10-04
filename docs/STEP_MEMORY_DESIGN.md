# Circular memory and refresh at simulated agent steps

This independent lab preserves the original double-circular-buffer idea while the
neural compressor is investigated separately. **It is a scripted CPU simulation,
not a learned memory model, real tool runner, or account of ChatGPT internals.**
An assistant's visible status updates do not demonstrate that it refreshes hidden
memory at those times.

## What it does

An environment streams randomly valued integer facts with opaque randomized keys,
independent of archive positions, an optional current-task
cue, simulated tool-step boundaries, corrections, distractors, and a final query.
The policy processes each event individually. It cannot inspect future events,
the final question before it arrives, or the evaluator's correct answer.

The fast ring holds four complete records. Evictions write `(key, archive address)`
breadcrumbs into a slower sixteen-slot ring. Both really overwrite their oldest
slots. Exact historical records remain in a separately charged append-only
archive. A key is explicit in a task cue or question: this is deliberately an
addressing and scheduling experiment, **not learned semantic retrieval**.

The normal policies have access only to their resident records and breadcrumbs.
The ordinary exact-lookup baseline also has a growing external key-to-latest-record
index. Environment bookkeeping tracks versions to create new records and score
answers; normal policies cannot query that dictionary. Its size is reported
separately so the simulation's Python allocations are not confused with the
modeled controller's storage.

For exact lookup, the environment-version index and policy index share the same
Python dictionary. They are separate logical roles; do not add their reported
sizes as though they were two physical dictionaries.

Nine policies cover dual-ring refresh at every eligible step, periodic steps,
question time, and no refresh; matched-budget single-ring variants; ordinary
indexed lookup; and a deliberately corrupted-pointer control. Step refresh uses
the currently visible task cue, while question-time refresh uses the now-visible
question. The default cap is **one lookup attempt and at most one record read per
episode**. The step policy continues to be invoked but cannot exceed that cap.
This deliberately small budget makes wasted prefetches visible; it is not a claim
that one read is an optimal production policy.

## Byte and work accounting

- Full record: four packed int64 fields, 32 bytes (key, value, version, address).
- Breadcrumb: two packed int64 fields, 16 bytes (key, address).
- Each ring: 16 bytes of cursor/occupancy metadata.
- Shared controller state: 24 bytes (current cue, step count, spent call budget).
- Dual ring: 416 bytes of rings plus 24 bytes of controller state = **440 bytes**.
- Single ring: twelve full records plus cursor metadata = 400 bytes, with a
  16-byte unused remainder under the same ring budget, plus the same controller.
- Archive: 32 bytes per streamed fact, with all versions retained.
- Exact baseline index: another 16 bytes per distinct key; therefore its total
  storage grows even though its active ring has the same fixed budget.

JSON reports allocated bytes, archive/index growth, actual lookup attempts,
records/bytes reread, answer-time calls, source ages, ring wraps, accuracy,
abstentions, stale-version errors, and exact-index probe payload bytes. Index
probes count a logical 16-byte entry per call, not measured hash-table traffic.
These are **logical packed sizes**, not
Python object memory or process RSS. Experiment counters and evaluator truth are
measurement/environment machinery, not hidden free memory available to a policy.
Each read injects one 32-byte record; there are no LLM tokens or KV caches here.
Source age counts subsequently archived records, not steps or elapsed time.

Equal *caps* do not guarantee equal *consumed* work: an absent cue or an expired
pointer may result in no read. The `matched_timing` stratum deliberately keeps
the cue valid and delay short, making the scheduled and on-demand dual policies
consume exactly the same one read. Other strata study policy failures under equal
caps and must not be represented as pure timing ablations.

## Reproduce

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_step_memory.py -v
.venv\Scripts\python.exe -m breadcrumb_memory.step_memory --output runs/step-memory --episodes 1000 --seed 17
```

The command writes `results.json` and `report.md` to a new or empty directory. No
network, model download, GPU, or external API is required. Ten unit tests cover
overwrite order, causal step boundaries, prefix invariance, version correctness,
hard/equal budgets, absent cues, wrong pointers, repeated turnover, reproducibility,
exactly matched timing reads, and opaque keys that do not encode archive addresses.

## Reviewed result (three seeds, 1,000 episodes per scenario per seed)

There are six scenario groups and nine policies, at seeds 17/29/43: 162,000
episode-policy evaluations. The table pools equal-sized seed sets. These opaque-key
runs supersede the preliminary sequential-key pilot.
Every episode begins with 64 facts, enough for repeated turnover of both rings.
Independent random delays include 0, 2, 8, or 24 new records. In the dedicated
timing stratum delays are restricted to 0 or 2 records.

| Scenario | Scheduled dual | Question-time dual | Exact indexed lookup |
|---|---:|---:|---:|
| Matched timing, stable early cue | 100% | 100% | 100% |
| Stable cue, variable delay | 50.2% | 62.1% | 100% |
| No early cue | 0% | 63.7% | 100% |
| Task changes after initial prefetch | 0% | 56.5% | 100% |
| New observed version after initial prefetch | 52.4% | 76.5% | 100% |
| Pointer already expired before cue | 0% | 0% | 100% |

The timing stratum has one archive read for both dual policies, but zero
answer-time calls for scheduled versus one for demand. This only demonstrates
moving known work earlier; **it does not measure latency savings or asynchronous
overlap**. Under these short stable delays, the single raw ring reaches 48.8% versus
dual 100%, and the wrong-pointer dual control reaches 0%. This is the predictable
benefit of keeping more compact addresses backed by a larger archive, not evidence
of a learned advantage over ordinary retrieval.

With longer delays, prefetched records can themselves be overwritten. Changed
tasks waste a scarce read. Without an early cue, the proactive policy has no
basis for selecting a record. If both resident content and breadcrumb have expired,
this mechanism cannot recover the item even though the archive still contains it.
The exact external index avoids that failure at a growing storage cost.

No stale-version outputs occurred: corrections are explicitly observed and the
reader prefers the latest retained address/version. This **does not test invisible
external mutations, cache-coherence races, or whether a model notices corrections**.
The corrupted-pointer control can still answer if a recent correction is already
in the fast ring; that is expected and should not be mistaken for failed corruption.

## Next useful experiment

Keep this deterministic controller as a systems baseline. Once a neural adapter
demonstrates actual entity-to-value binding, replace explicit keys with model
states and learn the cue/trigger policy. Sweep read budgets and delays, charge
actual injected tokens, and measure wall-clock latency with real retrieval and
model inference. Add sequential queries to test long-running budgets and refresh
competition. Compare fixed-size approximate breadcrumbs with an ordinary growing
index at matched total resource budgets.

The current finding supports a small, concrete engineering statement: bounded
active state can trade growing archive storage and read work for historical recall,
and useful early cues can move reads off the answer path. It establishes neither
constant total memory nor equivalence to a full-context language model.
