# Prompt-set patching

A patch that changes one prediction may fail on another example. Prompt sets show how the same residual intervention behaves across your supplied examples, including examples whose effects disagree with the mean.

## Run an evaluation

1. Load a checkpoint and open Residual patching in the Research lab.
2. Select Prompt set. Edit the example pairs or import a JSON array or JSONL file.
3. Supply clean and corrupted prompts, a target and an alternative answer. Each answer must be one token. Preserve leading spaces when the tokenizer requires them.
4. Choose logit difference or target log probability, then choose the last position, changed token positions or all positions. Advanced settings select physical block indices and the bootstrap seed.
5. Check the prompt set. Validation checks every pair before any model forward. Each pair needs equal nonzero token lengths, but separate pairs may have different lengths.
6. Run the set. Inspect a layer and expand individual pairs to check disagreement, source text and token positions. Save the result to the notebook or export JSON.

An imported record has these fields. The label and alternative are optional, but logit difference requires an alternative.

```json
[
  {
    "label": "Example one",
    "clean_prompt": "The capital of France is",
    "corrupted_prompt": "The capital of Poland is",
    "target": " Paris",
    "alternative": " Warsaw"
  }
]
```

These prompts illustrate the file format. Their token lengths and answer tokenization depend on the loaded tokenizer. The interface validates both.

## Read the measurements

For each layer, the effect is the patched score minus the corrupted score. Logit difference scores target logits minus alternative logits. Log probability scores the target token alone. The clean baseline gap is the clean score minus the corrupted score.

Raw summaries include the mean, median, minimum, maximum and positive, negative and near-zero fractions. Near-zero effects have absolute magnitude at most 0.00001. Individual examples remain inspectable even when their effects contradict the aggregate.

Normalized recovery divides the sum of effects by the sum of clean baseline gaps. It includes only examples whose clean gap exceeds 0.001. This is a ratio of sums, not the average of individual recovery ratios. The interface shows the excluded count. Small gaps can amplify recovery, so the raw effect remains the default view. Negative recovery and recovery above one are possible.

All aggregate layers use the same cohort of fully completed pairs. Stopping a sweep retains completed measurements, but a partially measured pair contributes to no aggregate layer. Partial measurements remain in the export for inspection.

## Uncertainty and reproducibility

Pointwise 95 percent percentile intervals use 2,000 paired bootstrap resamples. Raw effects resample completed pairs. Recovery resamples only its positive-gap cohort and recomputes the ratio of sums in each draw. Fewer than two eligible examples have no interval. Fewer than ten examples trigger a small-sample caution.

Intervals describe resampling of the supplied pairs. Correlated paraphrases are not independent evidence of generalization. The intervals do not adjust for examining many layers or selecting the most promising result. Report the prompts, corruption choice and scoring metric alongside any conclusion.

Saved records preserve the request, checkpoint revision, execution settings, target and alternative token IDs, prompt token IDs, patched positions, physical layer indices, seed, PCG64 generator, NumPy version and source fingerprint. Results reopen after the model is unloaded. Reproducibility across different numerical libraries or hardware is not guaranteed.

## Bounds and interpretation

A set supports up to 64 pairs, 128 selected layers and 2,048 total forward passes. Preflight estimates cache and forward memory against current server resources. Runtime memory can still vary with architecture and kernels.

Residual replacement tests an intervention at a block output. It does not identify a complete circuit or establish that a layer is necessary. Logit-difference and log-probability conclusions can differ. Compare controlled prompt sets rather than interpreting one convenient example as a general mechanism.

The scoring choices follow the methodological motivation in [Zhang and Nanda, Towards Best Practices of Activation Patching in Language Models, Metrics and Methods](https://arxiv.org/abs/2309.16042). Their study examines how patching conclusions depend on corruption and metric choices. The set-level summaries and bootstrap reporting here are implementation choices, not a reproduction of that paper.
