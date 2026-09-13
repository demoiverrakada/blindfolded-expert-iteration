# CP3-T1 temporal-study compute benchmark

Date: 2026-09-11  
Status: conditional; checkpoint remains open

## Outcome

The laptop can train the proposed 1.7B LoRA and can produce coherent batched
generations, but the frozen operating configuration has not passed. Training
was stable in speed and accelerator memory, while both training and decoding
recorded swap activity. Thinking-mode decoding also failed the completion
gate.

No model-organism training is authorized yet.

## Correctness

- Added memory-bounded vocabulary loss in `src/bei/chunked_loss.py`.
- Full-logit and chunked losses match on CPU for chunk sizes 1, 2, 4, and 20.
- Hidden-state and LM-head gradients match the full-logit calculation.
- Chunks with no supervised targets are skipped.
- The function fails closed when no shifted label is supervised.
- Repository tests: 65 passed.

## Sustained training result

Configuration:

- Qwen3-1.7B, bf16 MPS;
- LoRA rank 16/alpha 32 over all seven declared module families;
- sequence length 1,536;
- micro-batch 1, accumulation 8;
- two warm-up and 50 measured optimizer steps;
- 256-token vocabulary-loss chunks.

Result:

| Metric | Value |
| --- | ---: |
| Measured wall time | 1,756.59 s |
| Mean optimizer step | 35.13 s |
| Median optimizer step | 35.12 s |
| Min--max optimizer step | 34.57--36.13 s |
| Supervised throughput | 233.18 tokens/s |
| Driver-allocated MPS memory | 6.74 GB |
| Swap-ins during measured window | 4,360 pages |

The timing distribution is stable and shows no meaningful thermal slowdown.
At this rate, 4 million SDF supervised tokens take approximately 4.77 hours,
and a 298-step accumulation-8 expert-distillation run takes approximately
2.91 hours, excluding evaluation and checkpoint overhead.

The repeated synthetic benchmark sequence became trivial during optimization,
so its loss value is not an efficacy measurement. It remains a valid compute,
autograd, and memory-path exercise.

## Decode diagnostics

### SDPA, thinking, batch 16

- 94.73 aggregate generated tokens/s;
- 24.60 GB driver allocation;
- 0/16 closed the thinking block by 512 tokens;
- 15/16 padded rows collapsed into repeated punctuation.

This configuration is invalid and permanently excluded.

### SDPA, thinking, batch 4

- 26.05 aggregate generated tokens/s;
- 10.01 GB peak driver allocation;
- 0/16 closed the thinking block by 512 tokens;
- only the unpadded longest row in each batch decoded coherently.

The failure tracks left padding, not prompt content.

### Eager attention, thinking, batch 4

- 28.08 aggregate generated tokens/s at the 128-token diagnostic cap;
- 5.23 GB peak driver allocation;
- 16/16 coherent generations;
- 0/16 closed the thinking block.

This isolates the punctuation collapse to the MPS SDPA padded-batch path.
Eager attention is mandatory for future batched generation.

### Eager attention, non-thinking, batch 4

- 20.55 aggregate generated tokens/s;
- 10.12 GB peak driver allocation;
- 16/16 produced non-empty fenced answers;
- 11/16 terminated before 512 tokens;
- mean counted completion length 423.44 tokens;
- 63,840 swap-in and 184,816 swap-out pages were recorded.

The non-thinking fallback fixes output structure but requires a fresh template,
mask, and training-data audit. Five truncations out of sixteen are too many for
the current outcome protocol.

## Revised workload estimate

At the measured non-thinking rate, 200 prompts across nine 512-token-capped
conditions would require approximately 12.46 decode hours if every prompt used
the cap. Replay and reversal conditions add forward-pass and regeneration
costs, so this is a lower bound.

The temporal study remains possible inside the repository's 96-hour ceiling,
but only with:

- one gated organism-training path rather than the abandoned six-run masking
  design;
- eager attention;
- batch size at most four unless a fresh diagnostic validates another size;
- a frozen strategy for truncation and end-of-sequence handling;
- clean-memory swap remeasurement.

## Adversarial self-review

### Swap counts may include unrelated macOS activity

Yes. `vm_stat` is system-wide, not process-attributed. That prevents claiming
the model caused every page movement, but it also prevents claiming a
swap-free run. The correct response is a clean-memory repeat after closing
memory-heavy applications, not silently waiving the gate.

### The decode panel is simpler than BigCodeBench

Yes. It measures generation mechanics and formatting, not task accuracy.
Actual outcome prompts may be longer and slower. The workload projection is
therefore optimistic.

### Non-thinking changes the scientific object

Yes. It removes natural verbalized reasoning and may alter how evaluation
awareness is represented. The temporal experiment can still study hidden
state and generated text, but it cannot reuse thinking-mode masks, probes, or
interpretations without a new audit.

### Eager attention could alter numerics

Yes. Eager is not merely a throughput setting after SDPA visibly corrupts
padded generations. All training/evaluation comparisons must use one frozen
attention implementation, and a small unpadded eager-versus-SDPA agreement
test is required before treating eager outputs as the intended model.

## Requirements to close T1

1. Audit the non-thinking template and retokenize the candidate training set.
2. Add explicit EOS accounting and avoid counting batch padding as generation.
3. Repeat eager decoding at batch sizes 2 and 4 after a clean-memory reset.
4. Select a cap that leaves at most 5% truncation, then recompute the full
   temporal workload.
5. Repeat a shorter sustained training window under the same clean-memory
   conditions; require no swap-ins during the measured window.
6. Freeze the attention implementation, batch size, cap, and numeric budget
   before T2.

Failure of the clean-memory or truncation gate stops the local branch.
