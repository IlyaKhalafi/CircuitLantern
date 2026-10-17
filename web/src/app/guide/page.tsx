"use client";
import { useI18n } from "@/i18n/provider";
import {
  Activity,
  ArrowUpRight,
  Box,
  FlaskConical,
  GitCompareArrows,
  Layers3,
  Network,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import { PageHeader } from "@/components/ui";
export default function GuidePage() {
  const { t, rich } = useI18n();
  return (
    <>
      <PageHeader
        eyebrow={t("THE FIELD GUIDE")}
        title={t("Field guide")}
        description={t(
          "A practical guide to your first investigation and the measurements behind it.",
        )}
        actions={
          <Link className="button primary" href="/">
            {t("Choose a model")}
            <ArrowUpRight size={14} />
          </Link>
        }
      />
      <div className="guide-grid">
        <article className="panel guide-card">
          <div className="model-icon orange">
            <Box size={24} />
          </div>
          <h2>{t("Begin with a small checkpoint")}</h2>
          <p>
            {t(
              "SmolLM2 135M, GPT-2 and Pythia 70M are useful starting points. The tiny random GPT-2 checkpoint exercises the workflow quickly, but its untrained predictions carry no linguistic meaning.",
            )}
          </p>
          <p>
            {rich(
              "Select a checkpoint in the {v0}. Check the CPU or CUDA target, inspect the compatibility report, then download and open it.",
              {
                v0: <Link href="/">{t("model library")}</Link>,
              },
            )}
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon green">
            <ShieldCheck size={24} />
          </div>
          <h2>{t("Understand the deployment check")}</h2>
          <p>
            {t(
              "The backend measures this server, including available RAM, free GPU memory, free disk, GPU compute capability and installed packages. It uses the first CUDA device when GPU execution is selected.",
            )}
          </p>
          <p>
            {t(
              "Choose FP32, FP16, BF16, bitsandbytes NF4 or LLM.int8 execution. Kernel probes verify quantization support on the selected device. Matrix statistics use float32, dequantizing only the selected matrix. Native NVFP4 requires Blackwell hardware and its analysis adapter is still pending. An H200 lacks native NVFP4 execution.",
            )}
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon blue">
            <Network size={24} />
          </div>
          <h2>{t("Explore weight geometry")}</h2>
          <p>
            {t(
              "The architecture graph follows embeddings, transformer blocks and the unembedding. Pan and zoom, then select a block to narrow the matrix list. Select a matrix to analyze it.",
            )}
          </p>
          <ul>
            <li>
              {t(
                "Singular values describe the strength of principal linear directions.",
              )}
            </li>
            <li>
              {t(
                "Stable rank is the squared Frobenius norm divided by the squared spectral norm.",
              )}
            </li>
            <li>
              {t(
                "The spectral radius is the largest absolute eigenvalue, defined for square matrices.",
              )}
            </li>
            <li>
              {t(
                "The condition number compares the largest and smallest singular values.",
              )}
            </li>
          </ul>
        </article>
        <article className="panel guide-card">
          <div className="model-icon purple">
            <Activity size={24} />
          </div>
          <h2>{t("Trace a prediction")}</h2>
          <p>
            {t(
              "The activation lab captures a deterministic forward pass, then optionally generates a short greedy continuation. Choose Capture only to inspect the prompt without extra generation. Prompt tokens, residual norms, intermediate predictions and attention patterns come from the loaded checkpoint. In Attention capture options, choose a prompt token to inspect per-head routing from that position. The default is the final prompt token.",
            )}
          </p>
          <p>
            {t(
              "The logit lens applies final normalization and the model unembedding to intermediate states. Choose a token position to follow its prediction, vocabulary ID, confidence, entropy and residual norm across layers without another forward pass. These are diagnostic projections. Their confidence is not calibrated. The final logit lens projection uses the model's normalized output. Residual magnitude is measured before final normalization for supported block layouts.",
            )}
          </p>
          <p>
            {t(
              "Optionally track chosen target tokens before capturing. In the logit lens, select a prompt position and compare their log probabilities and logit difference across layers. Leading spaces matter when resolving a target to its vocabulary ID.",
            )}
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon orange">
            <Layers3 size={24} />
          </div>
          <h2>{t("Test a causal hypothesis")}</h2>
          <p>
            {t(
              "Run a baseline, then bypass a transformer block or zero an attention head. Block bypass replaces its output with its input residual. Head zeroing removes one head output before the attention output projection at every prompt and generated token.",
            )}
          </p>
          <p>
            {rich(
              "The lab overlays baseline residual norms when the prompt, checkpoint and execution settings match. The {v0} also restores clean residual outputs in a corrupted prompt across every block. Prompts must align by token position. Attention is descriptive. A focused attention pattern alone does not establish causal importance.",
              {
                v0: <Link href="/causal">{t("causal tracing workspace")}</Link>,
              },
            )}
          </p>
          <p>
            {rich(
              "The {v0} tests query heads independently across one layer or the model. It reports target log probability changes and divergence from the baseline next token distribution. Zero ablation can move activations outside their usual distribution. Results describe the tested prompt and execution precision. Cancel between forwards and export completed heads, including partial results. Select a head index to prepare its prompt and ablation in the activation lab. The recorded checkpoint must be loaded first.",
              {
                v0: (
                  <Link href="/causal/heads">
                    {t("head ablation workspace")}
                  </Link>
                ),
              },
            )}
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon green">
            <FlaskConical size={24} />
          </div>
          <h2>{t("Keep reproducible evidence")}</h2>
          <p>
            {t(
              "Save a trace, matrix analysis or completed sweep as an experiment, add observations and export the result as JSON. Available partial sweep results can be saved too. Records preserve the captured checkpoint and execution settings alongside their measurements.",
            )}
          </p>
          <p>
            {t(
              "Experiments are stored in local SQLite and can be reopened without keeping the model loaded. Retrying a sweep save returns its existing record and preserves edited notes. Keep prompts and settings fixed when comparing interventions. Search names, models and observations to find saved evidence. Open an activation trace and choose a reference to compare prompt measurements from matching checkpoints and execution settings. Saved weight analyses retain their captured matrix properties and charts after the checkpoint is unloaded.",
            )}
          </p>
          <p>
            {t(
              "Choose a reference matrix to compare the same weight across saved revisions or execution formats. Overlay singular values and energy, then inspect property changes. Estimates remain labeled and missing spectrum tails remain unavailable. Norm changes compare magnitudes and do not measure the distance between weight matrices.",
            )}
          </p>
          <p>
            {t(
              "Saved residual patch sweeps can compare positions when their prompt pair, target, checkpoint, execution settings and baseline scores match. Overlay patch effects and recovery, then export the source records with per block changes. Unmeasured blocks remain unavailable.",
            )}
          </p>
        </article>
        <article className="panel guide-card">
          <div className="model-icon blue">
            <Sparkles size={24} />
          </div>
          <h2>{t("Learn and test sparse features")}</h2>
          <p>
            {t(
              "Train a ReLU or TopK sparse autoencoder on document-separated residual activations. Inspect held out reconstruction, sparsity, activating contexts and feature annotations.",
            )}
          </p>
          <p>
            {t(
              "Ablate or amplify a decoded feature contribution while preserving the original reconstruction error. Compare predictions, evaluate reconstruction loss and measure downstream feature responses. Excluded token positions remain untouched.",
            )}
          </p>
          <p>
            {t(
              "Feature graphs report total ablation effects, including indirect paths. They are not direct attribution graphs from cross layer transcoders. Small local corpora and sparse reconstruction alone do not establish monosemanticity.",
            )}
          </p>
          <Link href="/features">
            {t("Open the feature lab")} <ArrowUpRight size={14} />
          </Link>
        </article>
      </div>
      <div className="guide-callout">
        <h2>{t("Read the measurement labels")}</h2>
        <p>
          {t(
            "Representation geometry compares layer CKA across prompt token positions, final residual token directions, and each block's update magnitude. CKA needs at least three positions with nonzero centered variance. Gray cells mark undefined comparisons. These are diagnostics within one prompt.",
          )}
        </p>
        <p>
          {t(
            "Small and bounded narrow matrices use complete spectra. Other matrices use randomized SVD on the full matrix and report up to 64 approximate components. Full matrix norms and descriptive statistics use every element. Heatmap cells contain region means. Interactive traces capture at most 128 prompt tokens. Spectral radius is computed only for square matrices up to 768 \u00D7 768.",
          )}
        </p>
        <article className="panel guide-card">
          <div className="model-icon blue">
            <GitCompareArrows size={24} />
          </div>
          <h2>{t("Compare base and finetuned checkpoints")}</h2>
          <p>
            {t(
              "Capture a reference, load the other checkpoint and measure actual weight distances, residual drift and prediction changes. Only one model needs to stay loaded.",
            )}
          </p>
          <p>
            {t(
              "The comparison checks architecture, prompt inputs and vocabulary alignment. Different execution settings are flagged, and saved reports preserve both checkpoints.",
            )}
          </p>
          <Link className="text-link" href="/workspace?view=compare">
            {t("Open checkpoint comparison")}
            <ArrowUpRight size={14} />
          </Link>
        </article>
      </div>
    </>
  );
}
