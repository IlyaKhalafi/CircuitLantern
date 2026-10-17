# Sparse features and feature circuits

[Back to Circuit Lantern](../README.md)

The Feature lab connects dictionary learning to measured interventions. Load a compatible checkpoint, supply documents, train a sparse autoencoder and inspect its strongest activating contexts. Use the same dictionary to test features on a prompt or build a graph of downstream ablation effects.

## Train dictionaries

Choose one transformer layer or every layer, a ReLU or TopK method, dictionary expansion and a training corpus. Width can range from one to 64 times the model dimension, up to 65536 features and the server's artifact limit. Put one document on each line or import a plain text file. The included demonstration corpus is small and editable.

Check training resources before starting. The report includes additional RAM and VRAM, artifact storage, unique training and validation tokens, document groups and the total requested optimizer steps. The language model stays resident while its dictionary trains. Layers are processed sequentially to bound activation and optimizer memory. The default server weight-file limit is 1 GiB per dictionary and can be changed with `LANTERN_MAX_DICTIONARY_BYTES`. Available memory and disk are checked independently of this limit.

Capture uses the residual stream immediately after each selected transformer block. The teacher runs in evaluation and inference mode with no cache, attention capture or language model output head during corpus collection. Its weights do not receive gradients.

Exact text duplicates are removed. Unicode normalization, whitespace collapse and case folding group text variants before a seeded split of whole document groups. Original text and tokenization stay unchanged. Validation positions whose complete model input prefix matches a training position are excluded and counted. This can reduce the held out token count or leave too few eligible tokens for training. Paraphrases and other overlapping templates are not deduplicated. Special tokens and the first token position are excluded by default. Inference and interventions honor the recorded exclusions.

Only training activations determine the RMS scale and initial geometric median bias. Validation documents never fit the preprocessing or optimizer. A small held out split can produce unstable quality estimates, so the interface reports its size rather than implying a large benchmark.

## Methods

The ReLU dictionary uses an untied encoder and decoder, a shared centering bias, squared reconstruction error per coordinate and a feature L1 penalty. Decoder columns have unit norm. Their parallel gradient components are removed before each Adam step, followed by renormalization. This follows the dictionary learning approach described in [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html).

TopK retains at most the chosen number of positive encoder preactivations. Its encoder begins as the transpose of the decoder, with no encoder bias. An AuxK loss trains inactive directions against the detached reconstruction residual. This draws on [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093).

These are interactive adaptations. Preprocessing uses one training RMS scalar on raw residual activations rather than per-token normalization. The default inactivity window is 4096 optimizer token exposures to make AuxK useful on small corpora. ReLU neuron resampling and exponential weight averaging are disabled. Each dictionary records the exact implementation, hyperparameters, seed, preprocessing, corpus, checkpoint revision and execution settings.

The dictionary seed controls initialization and optimizer sample order. A separate document split seed controls the held out groups and token sampling, so changing the dictionary seed can keep the evaluation corpus fixed. CUDA kernels and tied TopK scores may differ across devices or runtime versions. Repeated optimizer token exposures are reported separately from unique captured tokens.

## Read the evidence

Explained variance is one minus total reconstruction error divided by the error of the training mean on the same evaluation set. It can be negative. A zero baseline variance is reported as unavailable. Reconstruction MSE uses original residual units.

Pooled feature frequencies and activating contexts describe the complete captured corpus. Separate training and held out activation frequencies are also reported, and every activating example records its split. Held out reconstruction metrics describe only validation tokens. A feature with no activating examples stays unnamed until a researcher supplies an annotation. Sparse reconstruction alone does not establish monosemanticity. [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) provides broader context for feature interpretation and steering.

Dictionary records and weights remain available after unloading the language model. Annotations are separate from immutable scientific measurements. Unsaved annotation drafts remain with their feature while browsing the same dictionary. Safetensors weights and manifests have checksums. Startup recovery and publication use an interprocess artifact lock. Deleting a dictionary removes its weights and examples while retaining previously saved circuit measurements.

## Compare decoder directions

Open a saved dictionary, select a feature and expand **Related decoder directions**. Rank the other decoder columns by highest or lowest signed cosine. Follow a result to its activating contexts and annotations. This comparison runs on the CPU from verified saved weights and remains available after unloading the language model.

The comparison normalizes each column and accumulates in float64, using bounded chunks rather than a full pairwise matrix. It excludes the selected feature and zero directions. Equal scores are ordered by feature number. Inactive features are included, and displayed frequencies describe the recorded corpus. The JSON export retains the checkpoint revision, execution settings, capture point and checksums of the source weights and manifest.

The v2 directions export records a UTC measurement time and identifies CPU analysis separately from teacher inference. Source checksums and the original training execution are nested under `dictionary_source`. Busy artifact access returns a retryable response instead of queuing weight loads.

Decoder overlap describes geometry. It does not establish shared meaning, activation correlation or causal influence. In particular, negative cosine is not evidence that two features are semantic opposites. Anthropic's [April 2025 research update](https://transformer-circuits.pub/2025/april-update/index.html) discusses antipodal decoder pairs and differences between SAE and transcoder geometry.

## Test a prompt

Inspect features to see normalized SAE coefficients across prompt positions. Gray cells are excluded positions. Ranking and averages use eligible positions only. Prompt labels compare normalized document groups with the recorded corpus. A prompt without a matching document can still share phrases or prefixes with training text.

Ablation subtracts one decoded feature contribution from the original residual. Amplification adds another copy. Both retain the original reconstruction error in that decomposition. Re-encoding the changed residual can activate the same feature again because decoder directions need not be orthogonal. Excluded positions remain untouched.

Prediction effects compare a fixed target token against the unmodified baseline. The API also accepts an explicit target token ID. Intended and realized intervention norms expose rounding in FP16 and BF16 execution. A rounded-away intervention is not evidence that a feature is irrelevant.

Reconstruction evaluation replaces one eligible residual stream with its SAE reconstruction or the training mean, then measures next-token cross entropy on the prompt. Loss recovery can be negative or exceed one. It is undefined for a zero baseline gap or a prompt without an observed next token. The result records whether the document group belonged to training, validation or neither.

Open **Compare two features** to compare the selected feature with another dictionary feature on a prompt. The comparison reuses one baseline forward pass and shows both coefficient trajectories, decoder cosine, activation cosine, Pearson correlation and active-position overlap. Every activation statistic uses the dictionary's eligible token positions. The overlap is the active intersection divided by the active union, with activity defined as a positive coefficient. An empty union has no defined overlap. Pearson correlation is unavailable for fewer than two eligible positions or constant activity. Zero norm vectors have no defined cosine. The export records both feature IDs, position counts and the measurement convention.

These are descriptive prompt statistics. Neighboring tokens are dependent observations, and a short prompt can give unstable correlations. Decoder geometry and activation overlap do not establish shared meaning or causal influence. Use novel prompts and measured interventions to investigate a hypothesis.

Open **Set a feature coefficient** to set a constant decoded coefficient at the selected eligible prompt positions. Positive and negative values can change a direction even when its original coefficient is zero. This adapts the feature steering method in [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). For original coefficient `a`, chosen value `c`, decoder column `d` and recorded input scale `s`, the residual update is `s * (c - a) * d`. The original reconstruction error is retained.

Values here use normalized SAE coefficient units. The paper reports strengths relative to training maxima; the interface shows the pooled recorded corpus maximum as a reference. Re-encoding the changed residual may produce different activations because decoder directions need not be orthogonal. The result reports effects on the next-token distribution and records the coefficient, feature, positions and source identity.

Expand **Intervention precision** to compare the constructed float32 residual update with the realized change after conversion to the model residual dtype. Norm reductions use float64. A zero intended change has no defined relative rounding error.

Use **Export feature inspection** to retain the captured coefficients, tokens, predictions and any intervention effect. **Export reconstruction** saves the loss measurements and scoring scope. Both exports include a UTC capture time, the current inference execution identity, and a separate dictionary source with weight and manifest checksums, training settings, corpus fingerprint and original training execution identity. Exporting uses the captured result without another model forward and remains available after unloading the model while the page stays open. Save the JSON before leaving or reloading the page.

## Measure feature circuits

Select one dictionary per layer and a prompt. The lab chooses the most active features in each selected layer, ablates each separately, and measures changes in later features and the final prediction. The current interactive graph supports up to eight dictionaries and four candidate features per layer.

Each edge is a total downstream intervention effect. It can include indirect paths through other layers. Feature edges report changes in normalized coefficients, while output edges report log probability changes. Edge widths are scaled within each destination and the display can prune weaker edges. Exports retain every selected measurement.

This graph is a prompt-specific ablation study. It does not implement the gradient edge attribution algorithm from [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647) or the cross layer transcoder replacement model in [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). Those papers motivate feature-level causal investigation but use different methods for direct edges.

Completed graph measurements are saved automatically. Cancelling a sweep retains its completed ablations and marks the stored run incomplete. Cancelling training retains a dictionary only after at least one completed optimizer step and records the partial training state. Model hooks are removed before a job reports its terminal state.


Activation corpora retain the exact special-token IDs excluded during sampling. New captures include both the public special-token map and added tokens marked special. Feature interventions reuse the recorded exclusion set. Older dictionaries without that set keep their original public-map eligibility convention.
