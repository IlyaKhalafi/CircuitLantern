# Circuit Lantern

**An open workbench for understanding what happens inside a language model.**

Choose a Hugging Face checkpoint, check whether your server can run it, and move from model structure to weight geometry to live activation traces in one place.

Circuit Lantern runs locally with a Next.js frontend and a Python backend. Model weights and experiment records stay on your server.

## What you can explore

- **Hardware aware model selection** using available RAM, free VRAM, disk space, GPU compute capability and runtime support
- **Interactive architecture graphs** with pan, zoom, transformer block selection and a searchable weight inventory
- **Weight geometry** with pooled heatmaps, singular value spectra, cumulative energy, norms, spectral radius, stable rank, condition numbers, distributions and sparsity
- **Live activation traces** with prompt tokens, residual stream norms, layer predictions, next token logits and attention patterns
- **Causal block interventions** with transformer block bypass and a baseline residual overlay
- **Experiment notebooks** with local SQLite persistence, editable observations and portable JSON exports

## Run locally

You need Node.js 20.9 or newer, [uv](https://docs.astral.sh/uv/), and enough memory for the model you want to inspect. The default Python installation uses CPU PyTorch so the complete workflow also works without a GPU.

```bash
git clone https://github.com/IlyaKhalafi/CircuitLantern.git
cd CircuitLantern
make install
make dev
```

Open [localhost on port 3000](http://localhost:3000). The API runs on port 8000 and exposes its schema at [the API documentation](http://localhost:8000/docs).

Start with **SmolLM2 135M** or **Pythia 70M** for research. Use **Tiny random GPT-2** for a quick workflow check. The tiny checkpoint is untrained and its predictions do not have linguistic meaning.

## A first investigation

1. Select a model in the library and review its deployment report.
2. Download the checkpoint and explore the architecture graph.
3. Select a weight matrix and inspect its spectrum and distribution.
4. Open the activation lab and trace a short prompt.
5. Run the same prompt with one transformer block bypassed.
6. Save both traces as experiments and record your observations.

## Compatibility is more than memory

Checkpoint size alone is insufficient. Native NVFP4 execution requires Blackwell hardware. An H100 or H200 does not gain native FP4 support by having enough VRAM. Software kernels and the analysis runtime must also support the checkpoint format.

The current runtime loads unquantized Transformers causal language models from safetensors in float32. It reports packed quantized checkpoints as unsupported even when the GPU is eligible. It does not silently dequantize them or promise unsupported acceleration.

The memory report includes float32 weights, residual capture, bounded attention, a conservative KV cache allowance and runtime overhead. It reserves 15 percent of currently available memory. Estimates cannot guarantee that concurrent workloads will leave enough memory at load time. CUDA checks use the first GPU. Multi GPU sharding is not implemented.

See [NVIDIA NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) and [Transformers fine grained FP8 requirements](https://huggingface.co/docs/transformers/quantization/finegrained_fp8) for native hardware requirements.

## Measurement conventions

Full matrix norms and descriptive statistics use all elements. Spectral analysis uses exact SVD for matrices with dimensions up to 768 and randomized SVD on the full matrix for larger weights, retaining 64 components with four power iterations. Approximate spectral norms and stable ranks are labeled. Cumulative spectral energy uses the full matrix squared Frobenius norm as its denominator. Condition numbers are reported only for exact spectra. Heatmap cells are adaptive region means. Spectral radius is computed exactly for square matrices up to 768 × 768.

The logit lens applies final normalization and the model unembedding to intermediate hidden states. Its intermediate probabilities are diagnostic projections and should not be treated as calibrated confidence. The final residual magnitude is captured before normalization through a block hook for supported architectures. Interactive captures accept at most 128 prompt tokens. Attention patterns average heads, while per head entropy is reported separately.

## Configuration

Copy [.env.example](.env.example) to `backend/.env` as needed. Never commit access tokens.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LANTERN_DATA_DIR` | `data` | Experiment database directory |
| `LANTERN_MAX_MODEL_BYTES` | `21474836480` | Maximum checkpoint download size |
| `LANTERN_MAX_PROMPT_TOKENS` | `128` | Interactive prompt capture limit |
| `HF_TOKEN` | unset | Access to gated Hugging Face repositories |
| `LANTERN_BACKEND_URL` | `http://127.0.0.1:8000` | Backend address used by Next.js |

Hugging Face downloads use the standard Hub cache. Configure `HF_HOME` to move it. For CUDA execution, install a CUDA enabled PyTorch wheel in the backend environment that matches your server. The default lockfile intentionally selects the CPU index. The server page reports whether CUDA is actually available in Python.

Both services bind to loopback by default. This is a personal research workstation with no authentication. Use a private SSH tunnel for remote access. Add authentication and deployment controls before exposing it to an untrusted network.

## Development

```bash
make test
make check
make build
```

The backend separates Hub access, hardware discovery, compatibility, model execution, numerical analysis and experiment storage into services. Model operations are serialized to keep hooks and weights consistent. Downloads run as background jobs. Only safetensors checkpoints are accepted and remote model code is disabled.

Contributions should keep the researcher workflow focused, document measurement assumptions, and include checks for numerical or runtime changes. See [the architecture notes](docs/architecture.md).
