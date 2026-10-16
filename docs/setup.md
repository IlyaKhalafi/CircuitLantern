# Setup

[Back to Circuit Lantern](../README.md)

## Run locally

You need Node.js 20.9 or newer, [uv](https://docs.astral.sh/uv/), and enough memory for the model you want to inspect. The default Python installation uses CPU PyTorch so the complete workflow also works without a GPU.

```bash
git clone https://github.com/IlyaKhalafi/CircuitLantern.git
cd CircuitLantern
make install
make dev
```

For an NVIDIA server, use `make install-cuda` instead of `make install`. It uses uv to select a CUDA wheel for the installed driver and verifies device availability. Set `TORCH_BACKEND=cu128` when you need a specific supported wheel family. The development commands preserve the installed PyTorch build. Running `uv sync` directly restores the CPU lockfile, so rerun the CUDA installation target afterward. See [uv PyTorch backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection).

Open [localhost on port 3000](http://localhost:3000). The API runs on port 8000 and exposes its schema at [the API documentation](http://localhost:8000/docs).

Start with **SmolLM2 135M** or **Pythia 70M** for research. Use **Tiny random GPT-2** for a quick workflow check. The tiny checkpoint is untrained and its predictions do not have linguistic meaning.

## Compatibility is more than memory

Checkpoint size alone is insufficient. Native NVFP4 execution requires Blackwell hardware. An H100 or H200 does not gain native NVFP4 support by having enough VRAM. Software kernels and the analysis runtime must also support the checkpoint format.

The runtime accepts supported Transformers causal language models in safetensors. Choose FP32, FP16, BF16, bitsandbytes NF4 or LLM.int8 execution. Auto preserves a supported packed checkpoint format and otherwise selects FP32. Unquantized sources can be converted to NF4 or int8 while loading. Existing bitsandbytes checkpoints keep their quantization configuration. Native 4 bit checkpoints using BF16 compute require Ampere or newer when targeting CUDA. Other supported native compute settings are FP32 and FP16. Native FP8 and NVFP4 analysis adapters remain pending even on eligible hardware.

NF4 and bitsandbytes FP4 are different from native NVFP4. CPU and CUDA quantization paths are checked with real kernel probes. CUDA NF4 requires Pascal or newer, while LLM.int8 requires Turing or newer. CPU fallback execution reduces weight storage but does not guarantee faster inference. On-load CPU NF4 conversion retains canonical packed weights and uses float32 compute, avoiding the shape restrictions and mutable storage of the AVX512 BF16 repacker. Execution badges read the captured quantization configuration so FP4 and NF4 remain distinct in live and saved results. Older 4 bit records without codebook metadata receive a generic bitsandbytes label.

The memory report accounts for the selected weight format, unquantized embeddings and output heads, residual capture, bounded attention, logit and MLP buffers, a conservative KV cache allowance and temporary matrix analysis space. Source checkpoint size remains unchanged when selecting on-load quantization. The deployment report separately counts cached bytes and the remaining download. Disk checks budget the missing files while the configured checkpoint size limit still applies to the complete source. Cache credits require exact revision entries with matching known file sizes and are recalculated for each check. Logical parameter counts come from a model initialized on meta tensors, avoiding packed byte counts. GPU replacement estimates credit unique CUDA parameter and buffer storage owned by the active model because the loader releases those tensors first. Tied weights and storage views are counted once, packed parameters use physical storage bytes, and memory held by other workloads is not credited. Free memory and owned storage are sampled under the model operation lock. CPU and host staging budgets remain conservative because available RAM can already include reclaimable memory mapped checkpoint pages. Host tensor bytes are not added again. The report reserves 15 percent of the resulting memory budget. Concurrent workloads can change availability before loading. CUDA checks use the first device visible to PyTorch and respect CUDA_VISIBLE_DEVICES. Multi GPU sharding is not implemented.

On Linux, RAM budgets honor readable cgroup v1 and v2 memory constraints for the backend process and its parents. The budget considers shared parent usage and reclaimable inactive file cache, then applies the same memory reserve. A known limit with unreadable usage cannot claim available capacity. This prevents container deployments from being assessed using the host RAM total alone. See the [kernel memory controller documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).

See [Transformers bitsandbytes support](https://huggingface.co/docs/transformers/quantization/bitsandbytes), [bitsandbytes hardware compatibility](https://huggingface.co/docs/bitsandbytes/installation), [NVIDIA NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) and [Transformers fine grained FP8 requirements](https://huggingface.co/docs/transformers/quantization/finegrained_fp8) for native hardware requirements.

## Configuration

Copy [.env.example](../.env.example) to `backend/.env` as needed. Never commit access tokens.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LANTERN_DATA_DIR` | `data` | Experiment database directory |
| `LANTERN_MAX_MODEL_BYTES` | `21474836480` | Maximum checkpoint download size |
| `LANTERN_MAX_PROMPT_TOKENS` | `128` | Interactive prompt capture limit |
| `HF_TOKEN` | unset | Access to gated Hugging Face repositories |
| `LANTERN_BACKEND_URL` | `http://127.0.0.1:8000` | Backend address read by the Next.js server at runtime |

Hugging Face downloads use the standard Hub cache. Download storage checks measure that cache filesystem, separately from experiment database storage. Configure `HF_HOME` to move it. The default lockfile selects the CPU index. The CUDA installation target replaces only the environment PyTorch build using driver aware selection. The server page reports whether CUDA is actually available in Python.

Both services bind to loopback by default. This is a personal research workstation with no authentication. Use a private SSH tunnel for remote access. Add authentication and deployment controls before exposing it to an untrusted network.
