"use client";
import { useState } from "react";
import { Play } from "lucide-react";
import { Panel, ErrorNotice } from "@/components/ui";
import { useI18n } from "@/i18n/provider";
import { post } from "@/lib/api";
import {
  CrosscoderBudget,
  CrosscoderProgress,
  type CrosscoderTask,
  useSessionDraft,
} from "@/components/crosscoder-shared";
import {
  validInteger,
  validNumber,
  type CaptureSummary,
  type CrosscoderCapabilities,
  type CrosscoderJobResult,
  type CrosscoderPlan,
} from "@/lib/crosscoders";

export function CrosscoderTraining({
  reference,
  compared,
  capabilities,
  task,
  blocked,
}: {
  reference: CaptureSummary;
  compared: CaptureSummary;
  capabilities: CrosscoderCapabilities | null;
  task: CrosscoderTask<CrosscoderJobResult>;
  blocked: boolean;
}) {
  const { t } = useI18n();
  const [draft, setDraft, ready] = useSessionDraft(
    "lantern-crosscoder-training-draft",
    {
      name: "",
      expansion: "4",
      steps: "256",
      batch: "128",
      learningRate: "0.001",
      penalty: "0.05",
      seed: "0",
      device: "cpu",
    },
  );
  const [plan, setPlan] = useState<CrosscoderPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const maxExpansion = Math.min(
    64,
    Math.floor(65536 / Math.max(reference.input_dim, compared.input_dim)),
  );
  const valid =
    ready &&
    validInteger(draft.expansion, 1, maxExpansion) &&
    validInteger(draft.steps, 1, 20000) &&
    validInteger(draft.batch, 1, 512) &&
    validInteger(draft.seed, 0, 4294967295) &&
    validNumber(draft.learningRate, Number.MIN_VALUE, 0.1) &&
    validNumber(draft.penalty, 0, 10) &&
    (capabilities?.devices || ["cpu"]).includes(draft.device as "cpu" | "cuda");
  const unavailableDevice =
    ready &&
    capabilities != null &&
    !capabilities.devices.includes(draft.device as "cpu" | "cuda");
  const busy = task.busy || planning || blocked;
  function edit(field: keyof typeof draft, value: string) {
    setDraft((current) => ({ ...current, [field]: value }));
    setPlan(null);
  }
  function payload() {
    return {
      name: draft.name.trim() || t("Shared feature comparison"),
      capture_ids: [reference.id, compared.id],
      expansion: Number(draft.expansion),
      steps: Number(draft.steps),
      batch_size: Number(draft.batch),
      learning_rate: Number(draft.learningRate),
      l1_coefficient: Number(draft.penalty),
      seed: Number(draft.seed),
      device: draft.device,
    };
  }
  async function check() {
    setPlanning(true);
    setError(null);
    setPlan(null);
    try {
      setPlan(await post<CrosscoderPlan>("/crosscoders/plan", payload()));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setPlanning(false);
    }
  }
  return (
    <Panel
      title={t("3 · Train shared features")}
      subtitle={t(
        "One joint ReLU encoder reconstructs both retained activation banks with separate decoders.",
      )}
    >
      <div className="feature-form">
        <p className="muted">
          {t(
            "Training uses retained tensors and leaves source model weights unchanged. The teacher model can be unloaded first.",
          )}
        </p>
        <div className="feature-form-grid">
          <label>
            {t("Comparison name")}
            <input
              dir="auto"
              maxLength={100}
              value={draft.name}
              placeholder={t("Shared feature comparison")}
              disabled={busy}
              onChange={(event) => edit("name", event.target.value)}
            />
          </label>
          <label>
            {t("Training device")}
            <select
              aria-label={t("Training device")}
              value={draft.device}
              disabled={busy}
              onChange={(event) => edit("device", event.target.value)}
            >
              {unavailableDevice && (
                <option value={draft.device} disabled>
                  {t("Saved device unavailable")}
                </option>
              )}
              {(capabilities?.devices || ["cpu"]).map((device) => (
                <option key={device} value={device}>
                  {t(device === "cuda" ? "GPU" : "CPU")}
                </option>
              ))}
            </select>
          </label>
        </div>
        {unavailableDevice && (
          <p className="notice" role="status">
            {t(
              "The saved training device is unavailable on this server. Choose an available device before training.",
            )}
          </p>
        )}
        <details className="feature-advanced">
          <summary>{t("Advanced training settings")}</summary>
          <div className="feature-form-grid">
            {(
              [
                ["expansion", "Expansion factor", 1, maxExpansion, 1],
                ["steps", "Optimizer steps", 1, 20000, 1],
                ["penalty", "Sparsity penalty", 0, 10, 0.01],
                ["batch", "Batch size", 1, 512, 1],
                ["learningRate", "Learning rate", 0.000001, 0.1, 0.001],
                ["seed", "Random seed", 0, 4294967295, 1],
              ] as const
            ).map(([field, label, min, max, step]) => (
              <label key={field}>
                {t(label)}
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  disabled={busy}
                  value={draft[field]}
                  onChange={(event) => edit(field, event.target.value)}
                />
              </label>
            ))}
          </div>
        </details>
        <ErrorNotice message={error} />
        <div className="feature-toolbar">
          <button
            className="button primary"
            disabled={!valid || busy}
            onClick={() => {
              setError(null);
              void task.start("/crosscoders/train", payload());
            }}
          >
            <Play size={14} />
            {t("Train shared features")}
          </button>
          <button
            className="button secondary"
            disabled={!valid || busy}
            onClick={check}
          >
            {t("Check resources")}
          </button>
        </div>
        <CrosscoderBudget plan={plan} />
        <CrosscoderProgress task={task} />
      </div>
    </Panel>
  );
}
