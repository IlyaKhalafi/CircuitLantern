# Architecture

Circuit Lantern is a personal research workstation with two local processes.

```mermaid
flowchart LR
    Browser[Next.js workbench] --> Proxy[Same origin API proxy]
    Proxy --> API[FastAPI]
    API --> Hardware[Hardware discovery]
    API --> Hub[Hugging Face metadata]
    API --> Fit[Compatibility assessment]
    API --> Runtime[Serialized model runtime]
    Runtime --> Weights[Weight analysis]
    Runtime --> Captures[Hidden states and attention]
    API --> Store[SQLite experiments]
```

## Backend boundaries

`hardware.py` observes resources and runs tiny execution probes on the first visible CUDA device. It does not change device configuration or manage other workloads. `hub.py` resolves model metadata and immutable checkpoint revisions. `compatibility.py` is a pure decision function, making hardware requirements testable without a GPU or network.

`architecture.py` derives logical parameter and quantizable weight counts from a model on meta tensors. `execution.py` resolves precision, probes kernel support and builds loading options. It shares native four bit codebook and compute dtype validation with compatibility checks, including the BF16 GPU requirement. `weights.py` translates supported packed storage into logical matrices for analysis.

`runtime.py` owns one model and tokenizer. Its operation lock serializes loading, unloading, matrix analysis and activation capture. Download workers update bounded in memory job records. Weight statistics use a small cache that is cleared whenever the model changes.

`tracing.py` converts captured residual states, logit lens projections and attention tensors into portable measurements. The runtime releases capture buffers before continuation generation while keeping any intervention hooks active through generation.

`head_sweep.py` runs bounded independent attention head interventions using the verified output layouts in `interventions.py`. `scoring.py` shares finite logit checks, single token target resolution and float64 log probability scoring across research sweeps.

`metrics.py` holds numerical calculations and declares the difference between exact statistics and randomized full matrix spectral estimates. `store.py` uses short SQLite transactions with a connection per operation. A versioned migration adds artifact kinds and source job identity while preserving legacy trace JSON. Saved job IDs have a unique index so concurrent or retried saves return one record without overwriting notes. Current databases reopen without a write transaction. The save endpoint copies only a published research job result and uses its captured identity independently of the active model.

## Frontend boundaries

The App Router separates the library, architecture explorer, activation lab, experiments, server and field guide. Shared components cover the shell, panels, charts, heatmaps and trace results. API types and formatting utilities live in `src/lib`.

The frontend calls a same origin API path. A Next.js route handler forwards requests to the Python service so browser requests do not need a separate backend origin. It reads the backend address at runtime, allowing the same production build to target a different server without rebuilding.

Residual patching and head ablation sweeps run in the same serialized executor as model loading. A sweep captures clean block outputs once, then runs one restored block per corrupted forward pass. The model revision is checked before execution and cannot be replaced by another background load during the sweep. Head sweeps perform one unchanged baseline and one forward per selected query head, with at most 512 heads per job. Model session identity is checked alongside the checkpoint revision. Cancellation is cooperative between forwards, preserving completed results. Head failures retain earlier successful heads and publish a failed terminal state. Cancellation during a final successful forward does not relabel a complete result as partial. Browser job polling survives navigation and detects expired job IDs after a server restart.

Fonts are bundled with the frontend so rendering does not depend on third party font requests.

## Current limits

- One model and one execution device per backend process
- FP32, FP16, BF16, bitsandbytes NF4 and LLM.int8 execution
- Float32 statistics on loaded or dequantized logical matrices
- Native FP8 and NVFP4 adapters remain pending
- No execution of custom repository code
- Model download jobs are ephemeral and do not survive a backend restart
- Saved traces and research sweeps are durable in SQLite
- GPU eligibility does not imply packed quantization inspection support
- No authentication or multi tenant isolation

Large model execution, distributed workers and additional packed weight adapters should retain these boundaries rather than adding format specific behavior to the API routes or the interface.
