# Score contributions

[Back to Circuit Lantern](../README.md)

Score contributions show how the final prompt token's embedding input and block updates align with a captured answer. They complement the logit lens and help choose hypotheses to test through interventions.

## Investigate an answer

1. Load a checkpoint and capture a prompt in the Activation lab.
2. Open Score contributions in the trace results. The default compares the two highest-scoring next tokens.
3. To investigate specific answers, open Track target tokens before capturing. Every target must resolve to one vocabulary token. Leading spaces matter.
4. Choose an answer and an alternative. The view can compare every retained answer without another forward pass, including in a saved notebook after unloading the model.
5. Check the observed score, contribution sum and reconstruction error before interpreting a block. Sort by magnitude to find large positive and negative writes.
6. Use Test with patching on a block to prepare a controlled experiment. Supply your own corrupted prompt, check its tokens and run the patching experiment.

Captures retain up to eight tracked answers plus the actual top two predictions, deduplicated by vocabulary ID. Equal prediction logits are ordered by vocabulary ID. Other answer directions require a new capture. Raw-logit mode is available without an alternative, but its values are not probabilities and are not invariant to a common logit shift.

## What is decomposed

For the final prompt token, let the first block input be the embedding contribution and let each block update be its actual output minus its actual input. The embedding contribution includes any token, position and other processing before the first block.

For LayerNorm, each contribution is centered by its own mean, scaled by the final norm gain and divided by the denominator computed from the full final residual. For RMSNorm, contributions are scaled without centering. Known adapters distinguish a stored gain from a one-plus-weight gain, including Qwen 3.5. Normalization bias and output-head bias each appear once as separate rows.

The selected score is the answer logit minus the alternative logit. Each component projects onto that same output-head direction. Positive values point toward the answer, negative values toward the alternative. Large opposing writes can cancel, so the interface reports signed values rather than percentages of a possibly tiny total.

The normalization denominator is recomputed in float64 from the captured native residual. Scalar projections also use float64. Native model arithmetic can differ through normalization rounding, matrix multiplication and output casting. The exported record preserves these differences instead of forcing the sum to equal the observed score.

This is a frozen-normalization decomposition of an observed run. Removing a component would also change later computation and normalization. A large direct write is not an ablation effect, a necessity result or a discovered causal circuit.

The residual-readout interpretation follows the [TransformerLens analysis guide](https://transformerlensorg.github.io/TransformerLens/content/analysis_tools.html). Circuit Lantern projects captured checkpoint weights directly and checks the decomposition against the model's actual logits.

## Numerical health

The observed score comes from the actual model output. The sum comes from the recorded component projections. Reconstruction error is their absolute difference. Relative error uses the absolute observed score and is unavailable at zero.

The strict token tolerance uses an absolute term of 0.00001 plus 0.00001 times the larger of one and the actual token logit magnitude. A selected contrast uses the stricter of the summed recorded token tolerances and that same formula applied to the observed contrast. This prevents two individually large logits from hiding an unreliable small difference.

The separate native rounding allowance uses observed absolute dot-product magnitudes, head bias and execution precision. It accounts for output dtype and permitted TF32. It is a diagnostic heuristic, not a certified bound or a statistical interval. The interface exposes both tolerances. Large contrast errors block preparing an intervention hypothesis from that decomposition.

Exports retain per-token actual logits, projected readouts, component sums, stage errors, tolerances, native epsilon, TF32 allowance, normalization module, epsilon, gain convention, denominator and arithmetic dtypes. They also retain the original prompt, vocabulary IDs and checkpoint execution provenance. No raw attribution vectors are saved.

## From a write to a causal test

A prepared hypothesis keeps the clean prompt, answer IDs, source checkpoint and execution profile, selected physical block, original trace ID and diagnostic score values. It restores the clean block output at the final prompt token during a separate corrupted run.

The server validates the source profile, clean token IDs, answer IDs, position and selected block before any patching forward. Reloading the same checkpoint in the same execution profile can use a new model session. An incompatible checkpoint, tokenizer or profile blocks the linked experiment.

A linked hypothesis has one prompt pair. Adding examples or changing the source question requires choosing Use as independent experiment. The ordinary prompt-set workflow remains available for testing effects across examples. Attribution writes and patching effects measure different quantities.

Every preparation creates a new draft without deleting earlier experiments. Refreshing its URL resumes the same draft. Return to other draft restores the existing prompt-set workspace. Detaching preserves edits while removing the source binding from the next request.

## Availability

Verified adapters cover standard LayerNorm and common Llama, Mistral, Mixtral, Qwen and Gemma RMSNorm modules. The actual norm output and head input must agree with the adapter. Unknown norms, quantized or nonlinear heads, output softcaps or scaling, reused blocks and transforms between block boundaries remain unavailable for this method. Other trace measurements still work.

This view describes only the final captured prompt token. Trailing padding is rejected for attribution. Changing the logit lens position does not change this scope.

Capture uses bounded final-position vectors on the existing prompt forward. Its hooks and tensors are removed before continuation generation. The native capture allowance is 64 MiB. Saved scalar evidence remains readable without the checkpoint.
