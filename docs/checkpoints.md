# Compare model checkpoints

Open **Model explorer**, then **Compare checkpoints**. Capture a reference from the loaded base model, load the finetuned repository or another revision through the library, and compare with the loaded model. The library returns to the comparison workflow after loading.

A reference retains the exact prompt inputs, selected logical weight matrices, raw residual vectors at the final prompt token, final vocabulary logits and diagnostic layer predictions. Only one language model needs to remain loaded.

## Measurements

- Weight distance is the actual Frobenius norm of the aligned matrix difference.
- Relative change divides that distance by the reference norm.
- Cosine similarity measures directional agreement. Zero vectors give an undefined result.
- Residual drift uses actual block output vectors at the final prompt position.
- KL divergence and total variation use the full actual next-token vocabulary distribution.
- Intermediate predictions use a diagnostic logit lens with the loaded final normalization and output projection.

Selected minus reference is the sign convention. Quantized matrices are dequantized to float32 before comparison. Accumulation uses float64. These distances assume matching coordinates and weight names across checkpoints. They do not establish causal importance.

## Alignment and provenance

The backend requires matching architecture, block classes, residual width, attention configuration, MLP width, residual branch configuration, output vocabulary size and selected matrix shapes. It also requires identical prompt model inputs and vocabulary mappings. Repositories and revisions may differ.

Use matching execution settings for a checkpoint study. When precision, quantization, device or library versions differ, the report explicitly warns that changes combine checkpoint and execution effects. Both source checkpoints and execution settings remain in the export.

## Retained evidence

Choose up to 12 matrices per reference. The reference limit is 128 MiB of tensor evidence. Available RAM, GPU memory and disk space are checked with the model resident. The library holds up to 32 references and 128 comparison reports.

Reports can be reopened or exported without either model loaded. Deleting a reference frees its tensors while preserving completed comparison reports. Invalid files are isolated from valid evidence and can be removed through the interface.

## Block computation paths

In the architecture graph, choose **Block dataflow**. Focus on attention, the MLP or the full block. Blue arrows carry activations and gold arrows carry residual contributions. Matrix nodes open the actual logical weight analysis.

The diagram distinguishes fused QKV projections, grouped query attention projections, gated MLPs and NeoX parallel residual branches. Detailed layouts support the inspected GPT-2, GPT-NeoX, Llama, Mistral, Qwen2 and Qwen3 block classes. Other architectures retain the overview and matrix inventory.

This is a summarized inference computation diagram. Head reshaping, positional encoding and normalization are summarized. It is not a measured attribution graph.
