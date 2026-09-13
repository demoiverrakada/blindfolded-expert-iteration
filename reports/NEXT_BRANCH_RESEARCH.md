# Next-branch research decision

Date: 2026-09-11  
Decision: conditional candidate only; do not revive the failed matched-mask
experiment and do not claim novelty from a generic steering benchmark

## The question

When an intervention changes evaluation-aware behavior and is then removed,
what carries the change forward?

Three channels are possible:

1. **Visible-text channel:** steering changes early reasoning tokens, and those
   tokens condition the rest of the answer after steering stops.
2. **Hidden-cache channel:** steering changes the key/value cache even when the
   visible token prefix is held fixed.
3. **Ongoing-computation channel:** neither prefix is sufficient; the model
   recomputes the evaluation-conditioned policy from the original context, so
   the intervention must remain active.

The proposed experiment separates these channels instead of treating
"steering worked" as one undifferentiated result.

## Why the previous finding is insufficient

The frozen recovery study established that lexically neutral sentences inside
an awareness-bearing trace are often evaluation-conditioned in context. That
is a useful measurement warning, but it is not a top-tier standalone result.

Zhuang and Aranguri (2026) already report a nearby behavioral phenomenon:
removing an explicit evaluation-awareness sentence from a chain of thought
often causes evaluation-aware framing to regenerate. Their paper explicitly
points toward global representation-level suppression as future work.

The new project therefore asks a causal question their deletion intervention
cannot answer: whether regeneration is carried by visible reasoning, hidden
autoregressive state, or repeated inference from the unchanged evaluation
context.

## Prior-art cross-examination

| Work | Intervention | What it establishes | Missing comparison |
| --- | --- | --- | --- |
| Hua & Qin, *Steering Evaluation-Aware Language Models to Act Under Deployment* (2025) | Generation-time steering begins after the first sentence and remains active | A linear direction can alter verbalized awareness and evaluation-conditioned behavior | No switch-off, replay, or persistence curve |
| Zhuang & Aranguri, *Not All Eval Awareness is the Same* (2026) | Delete an awareness sentence and regenerate | Explicit awareness can regenerate from the remaining prefix/context | Text deletion does not separate hidden state from text or test activation suppression |
| Heidari et al., *Input-Only Suppression of Evaluation Awareness* (2026) | Prompt-token-only negative steering | Framing and trait inference can move while task behavior remains largely unchanged | No generated-token pulse or post-offset persistence |
| Cho et al., *PIXEL* (2025) | Optimize prompt-token positions and layers | Position-selective prompt steering can outperform uniform prompt steering | No within-generation switch-off or causal cache/text decomposition |
| Pinto, *Activation Steering Transfer to Agents* (2026) | Apply additive steering in matched chat and ReAct settings while re-encoding transcripts to exclude KV-cache contamination | Behavioral steering strength can change across interaction scaffolds even when the direction remains represented | No finite pulse, switch-off, replay decomposition, or evaluation-awareness policy |
| Huang et al., *GCAD* (2026) | Cache-conditioned decoding with repeated contrastive correction | Long-context behavior may need repeated cache-level correction | No finite pulse, clean-cache replay, or evaluation-awareness policy |
| Du et al., *When is Your LLM Steerable?* (2026) | Measure early steered states while intervention remains active | Early activation-space changes predict later behavior | Prediction is not persistence after intervention removal |

Searches through 2026-09-11 found no indexed evaluation-awareness study that:

- turns activation steering off during a single generation;
- holds visible generated tokens fixed while changing only the hidden cache;
- rebuilds a clean cache from steered text to isolate the visible-text channel;
- measures reversal or hysteresis of the evaluation-conditioned policy.

This is a scoped absence claim, not proof that no unpublished or unindexed
experiment exists. Pinto and GCAD make a generic demonstration of sustained
activation control less interesting, but neither performs the proposed
finite-pulse and clean-cache decomposition.

## The stronger experiment

Use an evaluation-conditioned local model organism and compare:

1. **Unsteered:** no intervention.
2. **Continuous:** deployment steering throughout generation.
3. **Free pulse:** steering for an early token window, then off with the
   steered cache retained.
4. **Text-only replay:** generate an early prefix under steering, rebuild the
   cache by replaying those same tokens without steering, then continue.
5. **Cache-only forced prefix:** force the same early tokens in both arms while
   constructing one cache with steering and one without, then continue
   unsteered.
6. **Reverse pulse:** early deployment steering followed by matched
   evaluation steering, and the reverse order.
7. **Random-direction controls:** identical schedules and vector norms.

Conditions 3--5 identify combined, visible-text, and hidden-cache persistence.
Condition 6 tests path dependence: equal integrated intervention with a
different order should not matter under a simple cumulative-dose account.

## Why either result teaches something

- **Text-only persistence:** evaluation-aware behavior becomes
  self-maintaining through ordinary language-model conditioning. Removing one
  sentence is weak because the broader generated plan is the mediator.
- **Cache-only persistence:** apparently ordinary continuations can inherit a
  latent evaluation-conditioned state not recoverable from visible text.
- **Continuous-only control:** the original evaluation cue is repeatedly
  recomputed; a one-shot representation edit is not a durable intervention.
- **Order-dependent reversal:** the policy has a commitment transition or
  hysteresis, not merely a linear dose response.
- **No effect beyond continuous steering:** the evaluation-awareness
  direction controls instantaneous expression but not a stable policy state.

Each branch distinguishes mechanisms that recommend different monitoring or
control strategies.

## Adversarial self-review

### Strongest objection: "continuous control is already expected"

Pinto shows that behavioral coupling can be reset by an agent scaffold even
when an injected direction remains present, and GCAD motivates repeated
cache-aware correction. The contribution must therefore be the causal
decomposition and the evaluation-awareness-specific commitment/reversal
curve, not the unsurprising fact that steering strength depends on context.

### Strongest objection: "cache persistence is architecturally trivial"

A steered prefix naturally leaves altered keys and values in the cache.
Reporting that alone would be weak. The cache-only condition must force
identical tokens, and the text-only condition must rebuild a clean unsteered
cache. The scientific result is their relative effect on final policy, not
that cached tensors differ.

### Strongest objection: "a 1.7B local analogue may not transfer"

Correct. The local study can establish a method and a result in a small
organism, not a claim about the released 49B Nemotron organism. The report
must distinguish:

- direct facts about Hua-Qin artifacts;
- local Qwen causal results;
- untested transfer predictions for the original organism.

The project stops if the local model does not acquire a robust
evaluation/deployment gap and a causally effective held-out direction.

### Strongest objection: "trajectory divergence makes timing incomparable"

Primary windows are defined by generated-token count and repeated with
sentence-boundary windows as a robustness analysis. Cache-only comparisons
force the same token prefix. Text-only comparisons replay exact token IDs.
Final behavior is paired by prompt and seed.

### Strongest objection: "the model may verbalize awareness without changing behavior"

Behavioral type-hint uptake is primary. Verbalized awareness and probe
projection are separate secondary outcomes. The study explicitly allows these
measures to dissociate.

## Go/no-go judgment

This branch is stronger than the matched-mask project as a measurement design.
A plain pulse-versus-continuous comparison is still not top-5% work because it
would not identify what carries persistence.

The branch remains a candidate only if it delivers the exact causal
text-versus-hidden-cache decomposition, an order-matched reversal test, and a
robust evaluation-aware behavioral effect. If hardware or organism limits
reduce it to ordinary continuous steering, stop and widen the search again.

It is not yet an empirical contribution. It earns continued work only through
the gates in `TEMPORAL_COMMITMENT_PROTOCOL.md`; failure to build a local
organism stops the project without reinterpretation.

## Sources

- Hua & Qin (2025), arXiv:2510.20487
- Cho et al. (2025), arXiv:2510.10205
- Huang et al. (2026), arXiv:2605.10664
- Du et al. (2026), arXiv:2606.11599
- Pinto (2026), arXiv:2607.09156
- Heidari et al. (2026), arXiv:2607.25907
- Zhuang & Aranguri (2026), arXiv:2608.27340
