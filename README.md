<p align="center">
  <a href="#run-locally">
    <img src="web/public/brand/lantern-icon.webp" alt="Circuit Lantern logo" width="180" height="180" />
  </a>
</p>

<h1 align="center">Circuit Lantern</h1>

<p align="center">
  <strong>Illuminate the inner workings of language models.</strong>
</p>

<p align="center">
  Your local lab for mechanistic interpretability.<br />
  Explore weights, trace activations and test what changes a prediction.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-111827?style=flat-square&amp;logo=nextdotjs&amp;logoColor=93c5fd" alt="Next.js" />
  <img src="https://img.shields.io/badge/Python-111827?style=flat-square&amp;logo=python&amp;logoColor=fbbf24" alt="Python" />
  <img src="https://img.shields.io/badge/Local_inference-111827?style=flat-square&amp;labelColor=111827&amp;color=fbbf24" alt="Local inference" />
</p>

<p align="center">
  <a href="#run-locally"><strong>Get started</strong></a> &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#what-you-can-investigate">Features</a> &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#inside-the-lab">Gallery</a> &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="docs/setup.md">Docs</a>
</p>

<p align="center">
  <sub>Light and dark themes</sub><br />
  <sub>English &nbsp;·&nbsp; فارسی &nbsp;·&nbsp; العربية &nbsp;·&nbsp; 日本語 &nbsp;·&nbsp; Français</sub>
</p>

<br />

<p align="center">
  <a href="#inside-the-lab">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/images/workbench-dark.png" />
      <source media="(prefers-color-scheme: light)" srcset="docs/images/workbench-light.png" />
      <img src="docs/images/workbench-light.png" alt="Circuit Lantern model library with hardware checks and model discovery" width="1200" />
    </picture>
  </a>
</p>

## What you can investigate

- **Model discovery.** Browse Hugging Face models by trending score, downloads, likes or recency. Narrow the results by author, date window and popularity.
- **Hardware fit.** Check RAM, VRAM, disk space, GPU architecture and execution kernels before downloading. Quantization support depends on the device and runtime, not just model size.
- **Weight geometry.** Follow attention, MLP and residual connections inside transformer blocks. Select logical matrices to inspect singular values, spectral radius, norms, rank, distributions and spectral energy.
- **Prompt traces.** Follow residual streams, intermediate predictions, vocabulary confidence, entropy and attention routing. Explore captured token positions without another forward pass. [Score contributions](docs/logit-attribution.md) show which block updates align with an answer, with checks against the actual logits.
- **Checkpoint changes.** Compare a base model with a finetuned version using retained weights, layer activations and final output distributions. Keep both revisions and execution settings with the results.
- **Sparse features.** Train ReLU or TopK sparse autoencoders from a text corpus. Inspect held out reconstruction, activating contexts and feature interventions, then save graphs of measured downstream effects.
- **Causal interventions.** Residual patching reports target score changes and normalized recovery. Attention head ablation reports target score changes and KL divergence. Repeated token probes investigate candidate induction heads. Completed measurements remain available when a sweep stops.
- **Evidence across prompts.** Patch a set of clean and corrupted examples, compare target and alternative logits, inspect contradictory effects and retain pointwise bootstrap intervals. [Prompt-set guide](docs/prompt-sets.md).
- **Research notebook.** Save traces, matrix analyses and sweeps with checkpoint revisions, execution settings and observations. Compare saved evidence and export the source records after unloading the model.

## Run locally

Requires Node.js 20.9+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/IlyaKhalafi/CircuitLantern.git
cd CircuitLantern
make install
make dev
```

For NVIDIA GPUs, use `make install-cuda` instead of `make install`.

Open [localhost on port 3000](http://localhost:3000). Start with **SmolLM2 135M** or **Pythia 70M**. The tiny random GPT-2 checkpoint is useful for checking the workflow, but its predictions have no linguistic meaning.

For Docker with explicit GPU access, run `python3 scripts/deploy.py --gpus 0 --port 9001`. Use `--gpus 0,1` to expose a device list. See [Docker setup](docs/setup.md#docker-deployment) for details.

## Inside the lab

![Attention dataflow showing Q, K, V projections, the output matrix and the residual connection](docs/images/block-dataflow-attention.png)

[Model overview](docs/images/model-explorer.png) · [MLP connections](docs/images/block-dataflow-mlp.png) · [Dark theme](docs/images/block-dataflow-dark.png)

Explore Qwen 3.5 with connected [Gated DeltaNet](docs/images/qwen35-deltanet-flow.png) and [full attention](docs/images/qwen35-attention-flow-dark.png) paths. These complete graph panels come from a loaded 4B checkpoint.

| Deployment checks | Weight geometry |
| --- | --- |
| [![Hardware and kernel compatibility report before model loading](docs/images/deployment-check.png)](docs/images/deployment-check.png) | [![Logical weight heatmap, singular values and matrix properties](docs/images/weight-analysis.png)](docs/images/weight-analysis.png) |
| Check the selected device, execution format and memory budget. | Explore structure and spectra without a wall of individual numbers. |
| **Residual streams** | **Causal comparisons** |
| [![Residual magnitudes, prediction uncertainty and token activation heatmap](docs/images/activation-lab.png)](docs/images/activation-lab.png) | [![Measured residual patch effects and recovery across transformer blocks](docs/images/causal-comparison.png)](docs/images/causal-comparison.png) |
| Follow a prompt through the model's captured internal states. | Compare patch positions against a shared baseline and retain the evidence. |
| **Sparse dictionaries** | **Feature circuits** |
| [![Trained sparse dictionary with held out metrics and activating contexts](docs/images/sparse-features.png)](docs/images/sparse-features.png) | [![Measured effects of feature ablations on downstream features and a target prediction](docs/images/feature-circuit.png)](docs/images/feature-circuit.png) |
| Inspect learned directions, examples and reconstruction quality. | Test a feature hypothesis through interventions and saved output effects. |
| **Hub discovery** | **Checkpoint comparison** |
| [![Hugging Face discovery with trending models and ranking controls](docs/images/hub-trending.png)](docs/images/hub-trending.png) | [![Base and instruct checkpoint comparison with weight deltas and residual drift](docs/images/checkpoint-comparison.png)](docs/images/checkpoint-comparison.png) |
| Find a checkpoint using familiar rankings and focused filters. | Measure weight, residual and prediction changes between model versions. |

Measurements shown here use trained SmolLM2 135M base and instruct checkpoints. Screenshots capture a normal viewport with the complete sidebar. Open any preview for the full image.

## Know what you are measuring

Intermediate logit lens predictions are diagnostic projections. Attention patterns describe routing and do not establish causal importance. Feature circuit edges measure downstream ablation effects, not direct causal connections. Checkpoint comparisons assume aligned model coordinates. Matrix analyses distinguish exact measurements from estimates, and missing results remain unavailable.

Execution supports FP32, FP16, BF16, bitsandbytes NF4, native bitsandbytes FP4 checkpoints and LLM.int8 on supported CPU or CUDA paths. Native FP8 and NVFP4 analysis adapters remain pending. NVFP4 requires Blackwell hardware as well as a compatible runtime.

## Go further

- [Setup](docs/setup.md) covers CUDA installation, configuration and remote access.
- [Score contributions](docs/logit-attribution.md) explains direct logit attribution, numerical checks and preparing causal tests.
- [Measurements](docs/measurements.md) explains capture conventions, interventions and comparisons.
- [Sparse features](docs/features.md) covers dictionary training, validation, interventions and research references.
- [Model explorer](docs/model-explorer.md) explains source-derived connections, hybrid mixers and coverage.
- [Checkpoint comparisons](docs/checkpoints.md) explains retained references, alignment checks, model drift and block dataflow.
- [Development](docs/development.md) covers checks and browser verification.
- [Architecture](docs/architecture.md) describes the service boundaries and current limits.

Contributions should keep the workflow focused, preserve measurement provenance and include useful verification.
