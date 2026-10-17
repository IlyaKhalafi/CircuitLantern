"use client";

import Link from "next/link";
import {
  ArrowLeft,
  Download,
  FlaskConical,
  Play,
  RefreshCw,
  Square,
} from "lucide-react";
import { CausalNavigation } from "@/components/causal-navigation";
import { InductionResults } from "@/components/induction-results";
import { SaveJobExperiment } from "@/components/save-job-experiment";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import { useInduction } from "@/hooks/use-induction";
import { useI18n } from "@/i18n/provider";
import { bytes, downloadJSON } from "@/lib/api";

export function InductionWorkbench() {
  const { t } = useI18n();
  const lab = useInduction();
  const { settings, capabilities } = lab;
  const result = lab.job?.result;
  const selectedHeads =
    capabilities?.layers
      .filter(
        (item) => !settings?.layers || settings.layers.includes(item.layer),
      )
      .reduce((sum, item) => sum + item.heads, 0) || 0;
  const forwards = settings
    ? settings.examples * (1 + Math.min(settings.causal_heads, selectedHeads))
    : 0;
  return (
    <>
      <PageHeader
        eyebrow={t("REPEATED TOKEN PROBES")}
        title={t("Follow a pattern. Test the heads.")}
        description={t(
          "Repeat token blocks, rank matching attention and measure candidate head effects.",
        )}
        actions={
          <>
            {lab.canCancel && (
              <button
                type="button"
                className="button secondary"
                onClick={lab.cancel}
                disabled={lab.cancelling}
              >
                <Square size={13} />
                {lab.cancelling ? t("Cancelling") : t("Cancel probe")}
              </button>
            )}
            <button
              className="button secondary"
              disabled={lab.loading || lab.busy}
              onClick={lab.retry}
            >
              <RefreshCw size={14} />
              {t("Refresh runtime")}
            </button>
            {result && (
              <button
                className="button secondary"
                onClick={() => downloadJSON(result, "induction-probe.json")}
              >
                <Download size={14} />
                {t("Export probe")}
              </button>
            )}
          </>
        }
      />
      <CausalNavigation />
      <Link className="induction-back" href="/causal/heads">
        <ArrowLeft size={14} />
        {t("Head ablation")}
      </Link>
      <ErrorNotice message={lab.error === result?.error ? null : lab.error} />
      {lab.loading ? (
        <Loading label={t("Connecting to model runtime")} />
      ) : !lab.runtime?.loaded ? (
        <EmptyState
          icon={<FlaskConical size={28} />}
          title={t("Explore repeated token attention.")}
          href="/"
          action={t("Load a model")}
        >
          {t(
            "Choose a checkpoint to look for candidate induction heads using synthetic token sequences.",
          )}
        </EmptyState>
      ) : !capabilities ? (
        lab.error || lab.busy ? null : (
          <Loading label={t("Checking probe capabilities")} />
        )
      ) : !capabilities.supported ? (
        <ErrorNotice
          message={
            capabilities.reason ||
            t("Repeated token probes are unavailable for this execution.")
          }
        />
      ) : (
        settings && (
          <Panel
            title={t("Design a repeated token probe")}
            subtitle={t(
              "Uniform distinct token blocks are repeated without retokenization. No text is generated.",
            )}
          >
            <form
              className="induction-form"
              onSubmit={(event) => {
                event.preventDefault();
                lab.check(true);
              }}
            >
              <div
                className="induction-sequence"
                role="img"
                aria-label={t("Repeated token probe structure")}
                dir="ltr"
              >
                {Array.from(
                  { length: Math.min(4, Math.max(0, settings.repeats || 0)) },
                  (_, index) => (
                    <div key={index}>
                      <span>{t("Block {v0}", { v0: index + 1 })}</span>
                      <strong>
                        {"A B C"} <span aria-hidden="true">···</span>
                      </strong>
                      <small>
                        {t("{v0} tokens", { v0: settings.block_length })}
                      </small>
                    </div>
                  ),
                )}
              </div>
              <p className="inline-note">
                {t(
                  "{v0} probes and up to {v1} candidate ablations use {v2} forward passes.",
                  {
                    v0: settings.examples,
                    v1: Math.min(settings.causal_heads, selectedHeads),
                    v2: forwards,
                  },
                )}
              </p>
              <details className="induction-advanced">
                <summary>{t("Probe settings")}</summary>
                <div className="induction-fields">
                  {(
                    [
                      [
                        "block_length",
                        "Tokens per block",
                        4,
                        Math.min(64, capabilities.vocabulary_pool_size || 64),
                      ],
                      ["repeats", "Repeats", 2, 4],
                      ["examples", "Probe examples", 1, 32],
                      ["causal_heads", "Candidate heads to ablate", 0, 12],
                      ["seed", "Random seed", 0, 4294967295],
                    ] as const
                  ).map(([key, label, min, max]) => (
                    <label key={key}>
                      {t(label)}
                      <input
                        className="form-input"
                        type="number"
                        aria-label={t(label)}
                        min={min}
                        max={max}
                        step={1}
                        value={
                          Number.isFinite(settings[key]) ? settings[key] : ""
                        }
                        disabled={lab.busy}
                        required
                        onChange={(event) =>
                          lab.update({ [key]: event.target.valueAsNumber })
                        }
                      />
                    </label>
                  ))}
                  <label>
                    {t("Layers to test")}
                    <select
                      aria-label={t("Layers to test")}
                      value={settings.layers?.[0] ?? "all"}
                      disabled={lab.busy}
                      onChange={(event) =>
                        lab.update({
                          layers:
                            event.target.value === "all"
                              ? null
                              : [+event.target.value],
                        })
                      }
                    >
                      <option value="all">{t("All tested layers")}</option>
                      {capabilities.layers.map((item) => (
                        <option value={item.layer} key={item.layer}>
                          {t("Layer {v0}", { v0: item.layer })}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <label className="induction-check">
                  <input
                    type="checkbox"
                    checked={settings.include_start_token}
                    disabled={lab.busy}
                    onChange={(event) =>
                      lab.update({ include_start_token: event.target.checked })
                    }
                  />
                  {t("Include the configured start token when available")}
                </label>
                <p className="inline-note">
                  {t(
                    "Set candidate ablations to zero for attention patterns only. All scores use queries after the first block.",
                  )}
                </p>
              </details>
              {lab.plan && (
                <div className="induction-plan" role="status">
                  <strong>
                    {lab.plan.compatible
                      ? t("Probe fits the current resource budget")
                      : t("This probe needs fewer resources")}
                  </strong>
                  <p>
                    {t("Additional RAM {v0} · GPU {v1} · saved evidence {v2}", {
                      v0: bytes(lab.plan.estimated_host_bytes),
                      v1: bytes(lab.plan.estimated_gpu_bytes),
                      v2: bytes(lab.plan.estimated_disk_bytes),
                    })}
                  </p>
                  <ErrorNotice message={lab.plan.blockers} />
                </div>
              )}
              <div className="induction-actions">
                <button
                  type="button"
                  className="button secondary"
                  disabled={lab.busy}
                  onClick={(event) => {
                    if (event.currentTarget.form?.reportValidity())
                      lab.check(false);
                  }}
                >
                  {t("Check resources")}
                </button>
                <button
                  type="submit"
                  className="button primary"
                  disabled={lab.busy || !selectedHeads}
                >
                  <Play size={14} />
                  {lab.checking
                    ? t("Checking resources")
                    : lab.busy
                      ? t("Running probe")
                      : t("Run probe")}
                </button>
              </div>
            </form>
          </Panel>
        )
      )}
      {lab.job && (
        <div className="job-progress" role="status" aria-live="polite">
          <div>
            <span>{t(lab.job.message)}</span>
            <strong>{Math.round(lab.job.progress)}%</strong>
          </div>
          <progress
            aria-label={t("Probe progress")}
            value={lab.job.progress}
            max={100}
          />
        </div>
      )}
      {result && (
        <div className="section-gap">
          {result.execution_id !== lab.runtime?.metadata?.execution_id && (
            <div className="notice info">
              {t(
                "These results belong to a previous model session. Their recorded model and settings are preserved.",
              )}
            </div>
          )}
          <InductionResults key={"result-" + lab.job?.id} result={result} />
          {lab.job && (
            <SaveJobExperiment
              key={"save-" + lab.job.id}
              jobId={lab.job.id}
              defaultName={"Repeated token probe · " + result.model_id}
              disabled={lab.busy}
            />
          )}
        </div>
      )}
      {!result && (
        <div className="guide-callout">
          <h2>{t("Evidence for a hypothesis")}</h2>
          <p>
            {t(
              "Prefix matching attention can identify candidate induction heads. It does not establish copying or a mechanism for general in-context learning. Head ablation measures signed effects on this synthetic probe distribution.",
            )}
          </p>
        </div>
      )}
    </>
  );
}
