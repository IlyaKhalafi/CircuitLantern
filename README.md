# Circuit Lantern

**A local lab for mechanistic interpretability.**

Load a Hugging Face model, check whether it fits your hardware, and explore how it computes. Built with Next.js and Python. Models and experiments stay on your server.

![Circuit Lantern model explorer](docs/images/model-explorer.png)

- **Check deployment fit** across RAM, VRAM, storage, GPU architecture and execution kernels.
- **Explore weights** through an interactive model graph, matrix spectra, norms and rank.
- **Trace predictions** with residual streams, logit lenses, attention and causal interventions.
- **Keep evidence** with saved experiments, comparisons, research notes and JSON exports.

## Get started

Requires Node.js 20.9+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/IlyaKhalafi/CircuitLantern.git
cd CircuitLantern
make install
make dev
```

For NVIDIA GPUs, use `make install-cuda` instead of `make install`.

Open [localhost on port 3000](http://localhost:3000) and start with **SmolLM2 135M**.

Supports FP32, FP16, BF16, bitsandbytes NF4 and LLM.int8. Native FP8 and NVFP4 adapters are pending.

[Setup](docs/setup.md) · [Measurements](docs/measurements.md) · [Development](docs/development.md) · [Architecture](docs/architecture.md)
