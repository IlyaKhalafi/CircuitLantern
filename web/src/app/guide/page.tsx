import {
  Activity,
  ArrowUpRight,
  Box,
  FlaskConical,
  Layers3,
  Network,
  ShieldCheck,
} from "lucide-react";
import Link from "next/link";
import { PageHeader } from "@/components/ui";

export const metadata = { title: "Field guide" };

export default function GuidePage() {
  return (
    <>
      <PageHeader
        eyebrow="THE FIELD GUIDE"
        title="Make the model a little less mysterious."
        description="A practical guide to your first investigation and the measurements behind it."
        actions={
          <Link className="button primary" href="/">
            Choose a model <ArrowUpRight size={14} />
          </Link>
        }
      />
      <div className="guide-grid">
        <article className="panel guide-card">
          <div className="model-icon orange">
            <Box size={24} />
          </div>
          <h2>Begin with a small checkpoint</h2>
          <p>
            SmolLM2 135M, GPT-2 and Pythia 70M are useful starting points. The
            tiny random GPT-2 checkpoint exercises the workflow quickly, but its
            untrained predictions carry no linguistic meaning.
          </p>
          <p>
            Select a checkpoint in the <Link href="/">model library</Link>.
            Check the CPU or CUDA target, inspect the compatibility report, then
            download and open it.
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon green">
            <ShieldCheck size={24} />
          </div>
          <h2>Understand the deployment check</h2>
          <p>
            The backend measures this server, including available RAM, free GPU
            memory, free disk, GPU compute capability and installed packages. It
            uses the first CUDA device when GPU execution is selected.
          </p>
          <p>
            Choose FP32, FP16, BF16, bitsandbytes NF4 or LLM.int8 execution.
            Kernel probes verify quantization support on the selected device.
            Matrix statistics use float32, dequantizing only the selected
            matrix. Native NVFP4 requires Blackwell hardware and its analysis
            adapter is still pending. An H200 lacks native NVFP4 execution.
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon blue">
            <Network size={24} />
          </div>
          <h2>Explore weight geometry</h2>
          <p>
            The architecture graph follows embeddings, transformer blocks and
            the unembedding. Pan and zoom, then select a block to narrow the
            matrix list. Select a matrix to analyze it.
          </p>
          <ul>
            <li>
              Singular values describe the strength of principal linear
              directions.
            </li>
            <li>
              Stable rank is the squared Frobenius norm divided by the squared
              spectral norm.
            </li>
            <li>
              The spectral radius is the largest absolute eigenvalue, defined
              for square matrices.
            </li>
            <li>
              The condition number compares the largest and smallest singular
              values.
            </li>
          </ul>
        </article>
        <article className="panel guide-card">
          <div className="model-icon purple">
            <Activity size={24} />
          </div>
          <h2>Trace a prediction</h2>
          <p>
            The activation lab captures a deterministic forward pass, then
            generates a short greedy continuation. Prompt tokens, residual
            norms, intermediate predictions and attention patterns come from the
            loaded checkpoint.
          </p>
          <p>
            The logit lens applies final normalization and the model unembedding
            to intermediate states. These are diagnostic projections. Their
            confidence is not calibrated. The final logit lens projection uses
            the model's normalized output. Residual magnitude is measured before
            final normalization for supported block layouts.
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon orange">
            <Layers3 size={24} />
          </div>
          <h2>Test a causal hypothesis</h2>
          <p>
            Run a baseline, then bypass a transformer block or zero an attention
            head. Block bypass replaces its output with its input residual. Head
            zeroing removes one head output before the attention output
            projection at every prompt and generated token.
          </p>
          <p>
            The lab overlays baseline residual norms when the prompt, checkpoint
            and execution settings match. The{" "}
            <Link href="/causal">causal tracing workspace</Link> also restores
            clean residual outputs in a corrupted prompt across every block.
            Prompts must align by token position. Attention is descriptive. A
            focused attention pattern alone does not establish causal
            importance.
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon green">
            <FlaskConical size={24} />
          </div>
          <h2>Keep reproducible evidence</h2>
          <p>
            Save a trace as an experiment, add observations and export the
            result as JSON. Records include the checkpoint revision, token IDs,
            intervention, device and measured activations.
          </p>
          <p>
            Experiments are stored in local SQLite and can be reopened without
            keeping the model loaded. Keep prompts and settings fixed when
            comparing interventions.
          </p>
        </article>
      </div>
      <div className="guide-callout">
        <h2>Read the measurement labels</h2>
        <p>
          Representation geometry compares layer CKA across prompt token
          positions, final residual token directions, and each block's update
          magnitude. CKA needs at least three positions with nonzero centered
          variance. Gray cells mark undefined comparisons. These are diagnostics
          within one prompt.
        </p>
        <p>
          Large matrices use randomized SVD on the full matrix to bound compute
          cost. The leading 64 spectral components are approximate. Full matrix
          norms and descriptive statistics use every element. Heatmap cells
          contain region means. Interactive traces capture at most 128 prompt
          tokens. Spectral radius is computed only for square matrices up to 768
          × 768.
        </p>
      </div>
    </>
  );
}
