"use client";
import { FieldLabel } from "@/components/help-tooltip";
import { useEffect, useId, useState } from "react";
import { HelpTooltip } from "@/components/help-tooltip";
import { Play, Square, Upload } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Metric, Panel } from "@/components/ui";
import { bytes, number, post } from "@/lib/api";
import { useBackgroundJob } from "@/lib/jobs";
import { useCompletedJob } from "@/hooks/use-completed-job";
import type {
  DictionaryPlan,
  FeatureCapabilities,
  TrainingJobResult,
} from "@/lib/features";

export function FeatureTraining({
  capabilities,
  onTrained,
}: {
  capabilities: FeatureCapabilities | null;
  onTrained: (id?: string) => void;
}) {
  const { t } = useI18n();
  const [name, setName] = useState("Residual dictionary");
  const [documents, setDocuments] = useState("");
  const [layer, setLayer] = useState("0");
  const [method, setMethod] = useState("topk");
  const [expansion, setExpansion] = useState(4);
  const [active, setActive] = useState(16);
  const [steps, setSteps] = useState(256);
  const [samples, setSamples] = useState(4096);
  const [seed, setSeed] = useState(0);
  const [splitSeed, setSplitSeed] = useState(0);
  const [penalty, setPenalty] = useState(0.001);
  const [excludeSpecial, setExcludeSpecial] = useState(true);
  const [excludeFirst, setExcludeFirst] = useState(true);
  const [plan, setPlan] = useState<DictionaryPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const task = useBackgroundJob<TrainingJobResult>(
    "lantern-feature-training-job",
  );
  useEffect(() => {
    if (capabilities?.example_documents)
      setDocuments(
        (current) => current || capabilities.example_documents.join("\n"),
      );
  }, [capabilities]);
  useCompletedJob(task, "lantern-feature-training-delivered", (result) => {
    onTrained(result.dictionaries[0]?.id);
  });
  useEffect(() => {
    setPlan(null);
  }, [capabilities?.execution?.execution_id]);
  useEffect(() => {
    if (capabilities?.input_dim) {
      const allowed = [1, 2, 4, 8, 16, 32, 64].filter(
        (value) => value * capabilities.input_dim! <= 65536,
      );
      setExpansion((current) =>
        allowed.includes(current) ? current : allowed[allowed.length - 1] || 1,
      );
      setActive((current) =>
        Math.max(1, Math.min(current, capabilities.input_dim! * expansion)),
      );
    }
  }, [capabilities?.input_dim, expansion]);
  function payload() {
    return {
      name,
      documents: documents.split(/\r?\n/).filter((text) => text.trim()),
      layers: layer === "all" ? capabilities?.layers : [Number(layer)],
      method,
      expansion,
      active_features: active,
      steps,
      max_samples: samples,
      seed,
      split_seed: splitSeed,
      l1_coefficient: penalty,
      exclude_special_tokens: excludeSpecial,
      exclude_first_position: excludeFirst,
      expected_execution_id: capabilities?.execution?.execution_id,
    };
  }
  async function check() {
    setPlanning(true);
    setError(null);
    setPlan(null);
    try {
      setPlan(await post<DictionaryPlan>("/features/plan", payload()));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setPlanning(false);
    }
  }
  async function upload(file?: File) {
    if (!file) return;
    if (file.size > 2_048_000) {
      setError("The corpus file must be smaller than 2 MB.");
      return;
    }
    try {
      setDocuments(await file.text());
      setPlan(null);
    } catch {
      setError("The corpus file could not be read.");
    }
  }
  const busy = task.busy || planning;
  const requirementId = useId();
  const requirement = !capabilities?.supported
    ? t("Load a supported model to train a dictionary.")
    : task.busy
      ? t("Training is running.")
      : planning
        ? t("Checking training resources")
        : !plan
          ? t("Check training resources before training.")
          : !plan.compatible
            ? t("Resolve the resource blockers below before training.")
            : null;
  return (
    <Panel
      title={t("Train a sparse dictionary")}
      help={t(
        "Learn sparse directions from the residual stream of your loaded model.",
      )}
    >
      <div className="feature-form">
        <ErrorNotice message={error || task.error} />
        <div className="feature-form-grid">
          <label>
            {t("Dictionary name")}
            <input
              className="form-input"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setPlan(null);
              }}
              maxLength={100}
              disabled={busy}
            />
          </label>
          <label>
            {t("Transformer layer")}
            <select
              aria-label={t("Transformer layer")}
              value={layer}
              onChange={(e) => {
                setLayer(e.target.value);
                setPlan(null);
              }}
              disabled={busy}
            >
              {(capabilities?.layers || []).map((index) => (
                <option value={index} key={index}>
                  {t("Layer {v0}", { v0: index })}
                </option>
              ))}
              <option value="all">{t("Every transformer layer")}</option>
            </select>
          </label>
        </div>
        <label>
          {t("Training corpus")}
          <textarea
            aria-label={t("Training corpus")}
            rows={7}
            value={documents}
            onChange={(e) => {
              setDocuments(e.target.value);
              setPlan(null);
            }}
            dir="auto"
            disabled={busy}
          />
        </label>
        <div className="feature-toolbar">
          <span className="control-label">
            <span>{t("One document per line.")}</span>
            <HelpTooltip label="Training corpus">
              {t(
                "Whole document groups are held out before tokens are sampled.",
              )}
            </HelpTooltip>
          </span>
          <label className="button secondary feature-upload">
            <Upload size={14} />
            {t("Import text file")}
            <input
              type="file"
              accept=".txt,text/plain"
              onChange={(e) => {
                void upload(e.target.files?.[0]);
                e.target.value = "";
              }}
              disabled={busy}
            />
          </label>
        </div>
        <details className="feature-advanced">
          <summary>{t("Advanced training settings")}</summary>
          <p className="inline-note">
            {t(
              "Defaults use TopK with AuxK. Tune the dictionary size, sparsity and training budget here.",
            )}
          </p>
          <div className="feature-form-grid">
            <div className="form-field">
              <FieldLabel
                id="dictionary-method"
                label="SAE method"
                help="TopK keeps a fixed number of active features. ReLU uses a sparsity penalty during training."
              />
              <select
                id="dictionary-method"
                aria-label={t("SAE method")}
                value={method}
                onChange={(e) => {
                  setMethod(e.target.value);
                  setPlan(null);
                }}
                disabled={busy}
              >
                <option value="topk">{t("TopK with AuxK")}</option>
                <option value="relu">{t("ReLU with L1")}</option>
              </select>
            </div>
            <label>
              {t("Dictionary expansion")}
              <select
                aria-label={t("Dictionary expansion")}
                value={expansion}
                onChange={(e) => {
                  setExpansion(Number(e.target.value));
                  setPlan(null);
                }}
                disabled={busy}
              >
                {[1, 2, 4, 8, 16, 32, 64]
                  .filter((n) => n * (capabilities?.input_dim || 1) <= 65536)
                  .map((n) => (
                    <option value={n} key={n}>
                      {n} ×
                    </option>
                  ))}
              </select>
            </label>
            {method === "topk" ? (
              <label>
                {t("Active features per token")}
                <input
                  className="form-input"
                  type="number"
                  min={1}
                  max={Math.min(
                    512,
                    (capabilities?.input_dim || 64) * expansion,
                  )}
                  value={active}
                  onChange={(e) => {
                    setActive(Number(e.target.value));
                    setPlan(null);
                  }}
                  disabled={busy}
                />
              </label>
            ) : (
              <label>
                {t("L1 coefficient")}
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  max={10}
                  step={0.0001}
                  value={penalty}
                  onChange={(e) => {
                    setPenalty(Number(e.target.value));
                    setPlan(null);
                  }}
                  disabled={busy}
                />
              </label>
            )}
            <label>
              {t("Training steps")}
              <input
                className="form-input"
                type="number"
                min={1}
                max={20000}
                value={steps}
                onChange={(e) => {
                  setSteps(Number(e.target.value));
                  setPlan(null);
                }}
                disabled={busy}
              />
            </label>
            <label>
              {t("Captured token limit")}
              <input
                className="form-input"
                type="number"
                min={3}
                max={32768}
                value={samples}
                onChange={(e) => {
                  setSamples(Number(e.target.value));
                  setPlan(null);
                }}
                disabled={busy}
              />
            </label>
            <label>
              {t("Random seed")}
              <input
                className="form-input"
                type="number"
                min={0}
                max={4294967295}
                value={seed}
                onChange={(e) => {
                  setSeed(Number(e.target.value));
                  setPlan(null);
                }}
                disabled={busy}
              />
            </label>
            <label>
              {t("Document split seed")}
              <input
                className="form-input"
                type="number"
                min={0}
                max={4294967295}
                value={splitSeed}
                onChange={(event) => {
                  setSplitSeed(Number(event.target.value));
                  setPlan(null);
                }}
                disabled={busy}
              />
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={excludeSpecial}
                onChange={(e) => {
                  setExcludeSpecial(e.target.checked);
                  setPlan(null);
                }}
                disabled={busy}
              />
              {t("Exclude special tokens")}
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={excludeFirst}
                onChange={(e) => {
                  setExcludeFirst(e.target.checked);
                  setPlan(null);
                }}
                disabled={busy}
              />
              {t("Exclude the first token position")}
            </label>
          </div>
        </details>
        <div className="feature-toolbar">
          <button
            className={plan?.compatible ? "button secondary" : "button primary"}
            disabled={busy || !capabilities?.supported}
            onClick={check}
          >
            {t("Check training resources")}
          </button>
          <button
            className={plan?.compatible ? "button primary" : "button secondary"}
            aria-describedby={requirement ? requirementId : undefined}
            aria-disabled={
              busy || !plan?.compatible || !capabilities?.supported
            }
            onClick={() => {
              if (!busy && plan?.compatible && capabilities?.supported)
                task.start("/features/train", payload());
            }}
          >
            <Play size={15} />
            {t("Train dictionary")}
          </button>
        </div>
        {requirement && (
          <p className="action-requirement" id={requirementId}>
            {requirement}
          </p>
        )}
        {plan && (
          <div className="feature-plan" role="status">
            <ErrorNotice
              message={plan.blockers.length ? plan.blockers : null}
            />
            {plan.warnings.map((warning) => (
              <p key={warning}>{t(warning)}</p>
            ))}
            <div className="metrics-grid">
              <Metric
                label="Training tokens"
                value={number(plan.unique_train_tokens, 0)}
              />
              <Metric
                label="Held out tokens"
                value={number(plan.unique_validation_tokens, 0)}
              />
              <Metric
                label="Additional RAM"
                value={bytes(plan.estimated_host_bytes)}
              />
              <Metric
                label="Additional VRAM"
                value={bytes(plan.estimated_gpu_bytes)}
              />
            </div>
            <p>
              {t("{features} features across {layers} layers", {
                features: plan.features,
                layers: plan.selected_layers.length,
              })}{" "}
              ·{" "}
              {t("{value} artifact storage", {
                value: bytes(plan.estimated_disk_bytes),
              })}
            </p>
            <p>
              {t(
                "{steps} optimizer steps and {forwards} language model forward passes",
                {
                  steps: number(plan.optimizer_steps_total, 0),
                  forwards: number(plan.teacher_forward_passes, 0),
                },
              )}
            </p>
            <p>
              {t(
                "{train} training document groups and {validation} held out document groups",
                {
                  train: plan.train_document_groups,
                  validation: plan.validation_document_groups,
                },
              )}
            </p>
            {(plan.validation_document_groups < 4 ||
              plan.unique_validation_tokens < 128) && (
              <p>
                {t(
                  "The held out split is small. More independent documents give a stronger quality estimate.",
                )}
              </p>
            )}
            {plan.truncated_documents > 0 && (
              <p>
                {t(
                  "{count} documents exceed the capture context and will be truncated",
                  { count: plan.truncated_documents },
                )}
              </p>
            )}
          </div>
        )}
        {task.job && (
          <div className="feature-job" role="status">
            <div className="feature-toolbar">
              <strong>{t(task.job.message)}</strong>
              <span>{number(task.job.progress, 0)}%</span>
            </div>
            <progress
              max={100}
              value={task.job.progress}
              aria-label={t("Dictionary training progress")}
            />
            {task.canCancel && (
              <button
                className="button secondary"
                onClick={task.cancel}
                disabled={task.cancelling}
              >
                <Square size={13} />
                {t(
                  task.cancelling
                    ? "Cancellation requested"
                    : "Cancel training",
                )}
              </button>
            )}
          </div>
        )}
        <p className="inline-note scientific-caveat">
          {t(
            "Reconstruction and sparsity do not prove that a feature has a single semantic meaning. Inspect examples and test interventions.",
          )}
        </p>
      </div>
    </Panel>
  );
}
