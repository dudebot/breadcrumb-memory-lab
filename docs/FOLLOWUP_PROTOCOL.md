# Follow-up protocol: bindings, two rings, and refresh between steps

Written before running the new experiments. These experiments stay in one repository but address separate claims.

## A. Does soft memory preserve relationships?

The existing 135M runs can improve by remembering which colors appeared without remembering their assignments. In the previous task there were two hidden values drawn from eight colors: a color-multiset-only strategy can average 56.25% forced-choice accuracy, higher than the observed trained accuracy. Therefore previous results do not establish object/color binding.

Use 256 fresh **quartets** at seed 34001 for each saved adapter, without retraining or selecting checkpoints:

1. Generate four distinct objects, two distinct old colors, and independent recent colors.
2. Encode the two old facts once, then ask about both old objects using the same memory.
3. Swap only the two old color assignments, preserving the color multiset, recent facts and question wording.
4. Encode this swapped history once and ask both questions again.

Primary metric: all four raw next-token answers correct. Also report per-answer accuracy, within-history both-queries accuracy, correct swapped pairs, restricted eight-color accuracy and candidate prediction changes. A decoder that ignores the query and always repeats one stored color cannot pass a quartet. Deterministic color-multiset-only memory cannot pass a quartet either.

Controls: full-text teacher, local text only, zero memory, canonicalized binding-blind memory (sort the old colors before encoding, so both assignments produce identical memory), and exact symbolic lookup. Keep all generated quartets; do not filter teacher errors. Separately report teacher-qualified quartets as a diagnostic with their denominator, not as the primary score.

Add teacher-only GPT-2 on the same formats and quartets. GPT-2 is a 124M comparison, not a scale-up. Probe its full-text competence before any memory training. Only train its compressor if it clears a separately sampled competence gate. If it fails, report the failure; do not compare its memory as though the teacher were equally capable.

Existing SmolLM2 outputs use absolute values of raw next-token accuracy, not conversational quality. All quartet outputs are greedy; randomized bag-of-colors guesses have different chance rates and are not the deterministic control.

## B. Does a second circular buffer help under a real budget?

Build a chronological simulated tool-event stream with a bounded fast workspace, compact slower breadcrumbs and an external evidence archive. The first implementation is a scripted harness, not a learned neural memory. This separates system behavior from the learned-compressor question.

The slower ring must genuinely overwrite, and the fast ring must lose old records. Compare single- and double-ring allocations at equal persistent byte ceilings, including logical metadata; report unused space caused by whole-record allocation. Report Python object overhead separately or explicitly exclude it from claims. Measure supporting-record survival, correct latest values, stale values, archive growth, index growth and reread volume.

## C. Does refresh timing help during an agent trajectory?

Refresh at step t may use only information available by t. A later final question must never be visible to the background refresh policy. Compare no refresh, periodic step-boundary refresh, cue-triggered step refresh and query-time retrieval under the same maximum calls and reread budget. Report actual use: equal ceilings alone do not mean equal consumed cost.

Include early useful cues, absent cues, changed tasks, stale facts and repeated ring turnover. Query-time retrieval gets the final question; background retrieval does not. Do not interpret that asymmetry as a fair accuracy-only contest: the potential benefit of prefetch is lower answer-time latency, which can trade off against wasted earlier reads.

User-facing progress messages are not evidence of an internal refresh operation in any proprietary model. A progress blurb is neither a necessary trigger nor an observable guarantee of that operation. The lab tests the design hypothesis directly through controlled simulated boundaries.

## Decision about scale

Do not scale model size merely because information binding or evidence routing fails. First isolate whether the bottleneck is teacher competence, the memory writer, the reader, the retrieval policy, or the benchmark. A 360M teacher is a reasonable later comparison if the smaller full-context teacher cannot solve the required task. No 7–8B run or GPU rental is needed for these follow-ups.

## Implementation-review amendment

After the first CPU pilot, independent review found sequential keys equal to
archive addresses. Final CPU runs use opaque randomized keys and supersede that
pilot. Before rerunning, the final set was fixed to seeds 17/29/43 with 1,000
episodes per scenario each. The budget wording above was clarified from equal
payloads to equal ceilings: the single ring leaves a 16-byte remainder. These
amendments were made after the pilot, not part of an untouched preregistration.
