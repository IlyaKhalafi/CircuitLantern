"use client";

import { useEffect, useId, useRef, useState } from "react";
import { Plus, Play, Square, Trash2, Upload } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { EmptyState, ErrorNotice, Loading, Panel } from "@/components/ui";
import { HelpTooltip } from "@/components/help-tooltip";
import { SaveJobExperiment } from "@/components/save-job-experiment";
import { PromptSetResults } from "@/components/prompt-set-results";
import { bytes, downloadJSON, post } from "@/lib/api";
import type { BackgroundJob } from "@/lib/jobs";
import {
  examplePairs,
  maxPromptPairs,
  PairImportError,
  parsePromptPairs,
  parseSetLayers,
  type PatchPositions,
  type PromptPair,
  type PromptSetMetric,
  type PromptSetPlan,
  type PromptSetResult,
} from "@/lib/prompt-sets";
import type { RuntimeState } from "@/lib/types";
import "./prompt-set.css";

interface Draft {
  pairs: PromptPair[];
  metric: PromptSetMetric;
  position: PatchPositions;
  layers: string;
  seed: string;
}
const initialDraft: Draft = {
  pairs: examplePairs,
  metric: "logit_difference",
  position: "last",
  layers: "",
  seed: "0",
};
const storageKey = "lantern-prompt-set-draft";

export function PromptSetWorkbench({
  runtime,
  loading,
  runtimeError,
  onBusyChange,
  task,
}: {
  task: BackgroundJob<PromptSetResult>;
  runtime: RuntimeState | null;
  loading: boolean;
  runtimeError: string | null;
  onBusyChange: (busy: boolean) => void;
}) {
  const { t } = useI18n();
  const [draft, setDraft] = useState<Draft>(initialDraft);
  const [ready, setReady] = useState(false);
  const [plan, setPlan] = useState<PromptSetPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [importing, setImporting] = useState(false);
  const [openPairs, setOpenPairs] = useState<Set<number>>(new Set([0]));
  const [error, setError] = useState<string | null>(null);
  const planSequence = useRef(0);
  const importRef = useRef<HTMLInputElement>(null);
  const requirementId = useId();
  const result = task.job?.result;
  const execution = runtime?.metadata?.execution_id;
  const busy = task.busy || planning || importing;
  useEffect(() => {
    onBusyChange(planning || importing);
  }, [planning, importing, onBusyChange]);
  useEffect(() => () => onBusyChange(false), [onBusyChange]);
  useEffect(() => {
    try {
      const stored = JSON.parse(sessionStorage.getItem(storageKey) || "null");
      if (
        stored &&
        ["logit_difference", "log_probability"].includes(stored.metric) &&
        ["last", "changed", "all"].includes(stored.position)
      )
        setDraft({
          pairs: parsePromptPairs(JSON.stringify(stored.pairs), {
            allowIncomplete: true,
          }),
          metric: stored.metric,
          position: stored.position,
          layers: typeof stored.layers === "string" ? stored.layers : "",
          seed: typeof stored.seed === "string" ? stored.seed : "0",
        });
    } catch {
      /* A private or full browser store must not block research. */
    }
    setReady(true);
  }, []);
  useEffect(() => {
    if (ready) {
      try {
        sessionStorage.setItem(storageKey, JSON.stringify(draft));
      } catch {
        /* Draft persistence is optional. */
      }
    }
  }, [draft, ready]);
  useEffect(() => {
    planSequence.current++;
    setPlan(null);
    setPlanning(false);
  }, [execution]);
  function edit(changes: Partial<Draft>) {
    planSequence.current++;
    setPlanning(false);
    setDraft((current) => ({ ...current, ...changes }));
    setPlan(null);
    setError(null);
    task.clearError();
  }
  function editPair(index: number, changes: Partial<PromptPair>) {
    edit({
      pairs: draft.pairs.map((pair, i) =>
        i === index ? { ...pair, ...changes } : pair,
      ),
    });
  }
  function payload() {
    const pairs = parsePromptPairs(JSON.stringify(draft.pairs));
    const layers = parseSetLayers(draft.layers);
    if (!/^\d+$/.test(draft.seed) || Number(draft.seed) > 4294967295)
      throw new Error("Use a whole-number seed between 0 and 4294967295.");
    return {
      pairs,
      layers,
      metric: draft.metric,
      position_mode: draft.position,
      seed: Number(draft.seed),
      expected_execution_id: execution,
    };
  }
  function describeError(reason: unknown) {
    setError(
      reason instanceof PairImportError
        ? t(reason.message, { count: reason.row })
        : (reason as Error).message,
    );
  }
  async function check() {
    const sequence = ++planSequence.current;
    setError(null);
    setPlanning(true);
    setPlan(null);
    try {
      const value = await post<PromptSetPlan>("/causal/sets/plan", payload());
      if (sequence === planSequence.current) setPlan(value);
    } catch (reason) {
      if (sequence === planSequence.current) describeError(reason);
    } finally {
      if (sequence === planSequence.current) setPlanning(false);
    }
  }
  async function importPairs(file: File | undefined) {
    if (!file) return;
    setImporting(true);
    try {
      if (file.size > 2 * 1024 * 1024)
        throw new Error("Choose a prompt-set file smaller than 2 MiB.");
      const pairs = parsePromptPairs(await file.text());
      edit({ pairs });
      setOpenPairs(new Set([0]));
    } catch (reason) {
      describeError(reason);
    } finally {
      setImporting(false);
    }
  }
  const canRun =
    !!runtime?.loaded &&
    !!execution &&
    !!plan?.compatible &&
    plan.execution_id === execution &&
    !busy &&
    ready;
  const requirement = !plan
    ? t("Check the prompt set before running.")
    : !plan.compatible
      ? t("Resolve the validation and resource blockers before running.")
      : null;
  return (
    <>
      <ErrorNotice
        message={
          runtimeError ||
          error ||
          (task.error === result?.error ? null : task.error)
        }
      />
      {loading ? (
        <Loading />
      ) : !runtime?.loaded ? (
        <EmptyState
          icon={<Play />}
          title={t("Load a model to evaluate a prompt set")}
          href="/"
          action="Open model library"
        >
          {t(
            "Saved prompt-set evidence remains available without a loaded model.",
          )}
        </EmptyState>
      ) : (
        <Panel
          title={t("Prompt-set patching")}
          help={t(
            "Test whether a patch effect repeats across examples. Every pair is validated before any model forward, and every layer is summarized over the same completed pairs.",
          )}
        >
          <form
            className="set-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (!busy && execution && ready) {
                if (canRun) {
                  try {
                    void task.start("/causal/sets/run", payload());
                  } catch (reason) {
                    describeError(reason);
                  }
                } else void check();
              }
            }}
          >
            <div className="set-form-controls">
              <label>
                {t("Scoring metric")}
                <select
                  aria-label={t("Scoring metric")}
                  value={draft.metric}
                  disabled={busy}
                  onChange={(event) =>
                    edit({ metric: event.target.value as PromptSetMetric })
                  }
                >
                  <option value="logit_difference">
                    {t("Target versus alternative")}
                  </option>
                  <option value="log_probability">
                    {t("Target log probability")}
                  </option>
                </select>
              </label>
              <label>
                {t("Positions to restore")}
                <select
                  aria-label={t("Positions to restore")}
                  value={draft.position}
                  disabled={busy}
                  onChange={(event) =>
                    edit({ position: event.target.value as PatchPositions })
                  }
                >
                  <option value="last">{t("Last prompt token")}</option>
                  <option value="changed">
                    {t("Changed token positions")}
                  </option>
                  <option value="all">{t("All token positions")}</option>
                </select>
              </label>
              <HelpTooltip label="Scoring metric">
                {t(
                  "Contrastive scoring subtracts the alternative token logit from the target token logit. Each requested answer must encode to exactly one token. Leading spaces are preserved.",
                )}
              </HelpTooltip>
            </div>
            <div className="set-pair-toolbar">
              <strong>{t("Prompt pairs")}</strong>
              <span>
                {t("{count, plural, one {# pair} other {# pairs}}", {
                  count: draft.pairs.length,
                })}
              </span>
              <button
                type="button"
                className="button secondary"
                disabled={busy}
                onClick={() => importRef.current?.click()}
              >
                <Upload size={14} />
                {t("Import prompt pairs")}
              </button>
              <input
                ref={importRef}
                type="file"
                hidden
                aria-label={t("Import prompt pairs")}
                accept=".json,.jsonl,application/json"
                disabled={busy}
                onChange={(event) => {
                  void importPairs(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />

              <button
                className="button secondary"
                type="button"
                onClick={() => downloadJSON(draft.pairs, "prompt-pairs.json")}
              >
                {t("Export pairs")}
              </button>
            </div>
            <div className="set-pairs">
              {draft.pairs.map((pair, index) => {
                const problem = plan?.pair_errors.find(
                  (item) => item.index === index,
                );
                return (
                  <details
                    key={index}
                    className={problem ? "set-pair invalid" : "set-pair"}
                    open={openPairs.has(index) || !!problem}
                    onToggle={(event) => {
                      const open = event.currentTarget.open;
                      setOpenPairs((current) => {
                        if (current.has(index) === open) return current;
                        const next = new Set(current);
                        if (open) next.add(index);
                        else next.delete(index);
                        return next;
                      });
                    }}
                  >
                    <summary>
                      <bdi>
                        {pair.label || t("Pair {count}", { count: index + 1 })}
                      </bdi>
                      {problem && <span>{t("Needs correction")}</span>}
                    </summary>
                    <div className="set-pair-fields">
                      <div className="set-pair-name">
                        <label>
                          {t("Pair label")}
                          <input
                            className="form-input"
                            value={pair.label}
                            dir="auto"
                            maxLength={100}
                            disabled={busy}
                            onChange={(event) =>
                              editPair(index, { label: event.target.value })
                            }
                          />
                        </label>
                        <button
                          type="button"
                          className="button secondary"
                          disabled={busy || draft.pairs.length === 1}
                          aria-label={t("Remove pair {count}", {
                            count: index + 1,
                          })}
                          onClick={() => {
                            edit({
                              pairs: draft.pairs.filter((_, i) => i !== index),
                            });
                            setOpenPairs(
                              (current) =>
                                new Set(
                                  [...current]
                                    .filter((i) => i !== index)
                                    .map((i) => (i > index ? i - 1 : i)),
                                ),
                            );
                          }}
                        >
                          <Trash2 size={14} />
                          {t("Remove")}
                        </button>
                      </div>
                      <div className="set-prompt-grid">
                        <label>
                          {t("Clean prompt")}
                          <textarea
                            aria-label={t("Clean prompt")}
                            dir="auto"
                            value={pair.clean_prompt}
                            maxLength={8000}
                            disabled={busy}
                            onChange={(event) =>
                              editPair(index, {
                                clean_prompt: event.target.value,
                              })
                            }
                          />
                        </label>
                        <label>
                          {t("Corrupted prompt")}
                          <textarea
                            aria-label={t("Corrupted prompt")}
                            dir="auto"
                            value={pair.corrupted_prompt}
                            maxLength={8000}
                            disabled={busy}
                            onChange={(event) =>
                              editPair(index, {
                                corrupted_prompt: event.target.value,
                              })
                            }
                          />
                        </label>
                      </div>
                      <div className="set-prompt-grid">
                        <label>
                          {t("Target token")}
                          <input
                            className="form-input"
                            dir="auto"
                            value={pair.target}
                            maxLength={100}
                            disabled={busy}
                            onChange={(event) =>
                              editPair(index, { target: event.target.value })
                            }
                          />
                        </label>
                        {draft.metric === "logit_difference" && (
                          <label>
                            {t("Alternative token")}
                            <input
                              className="form-input"
                              dir="auto"
                              value={pair.alternative}
                              maxLength={100}
                              disabled={busy}
                              onChange={(event) =>
                                editPair(index, {
                                  alternative: event.target.value,
                                })
                              }
                            />
                          </label>
                        )}
                      </div>
                      {problem && <ErrorNotice message={problem.message} />}
                    </div>
                  </details>
                );
              })}
            </div>
            <button
              type="button"
              className="button secondary set-add"
              disabled={busy || draft.pairs.length >= maxPromptPairs}
              onClick={() => {
                setOpenPairs(
                  (current) => new Set([...current, draft.pairs.length]),
                );
                edit({
                  pairs: [
                    ...draft.pairs,
                    {
                      label: "",
                      clean_prompt: "",
                      corrupted_prompt: "",
                      target: "",
                      alternative: "",
                    },
                  ],
                });
              }}
            >
              <Plus size={14} />
              {t("Add prompt pair")}
            </button>
            <details className="feature-advanced">
              <summary>{t("Layers and reproducibility")}</summary>
              <div className="set-prompt-grid">
                <label>
                  {t("Layer indices")}
                  <input
                    className="form-input"
                    dir="ltr"
                    placeholder={t("All layers, or indices such as 0, 4, 8")}
                    value={draft.layers}
                    disabled={busy}
                    onChange={(event) => edit({ layers: event.target.value })}
                  />
                </label>
                <label>
                  {t("Bootstrap seed")}
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={4294967295}
                    step={1}
                    value={draft.seed}
                    disabled={busy}
                    onChange={(event) => edit({ seed: event.target.value })}
                  />
                </label>
              </div>
            </details>
            <div className="feature-toolbar">
              <button
                type="button"
                className={canRun ? "button secondary" : "button primary"}
                disabled={busy || !ready || !execution}
                onClick={() => void check()}
              >
                {planning ? t("Checking prompt set") : t("Check prompt set")}
              </button>
              <button
                type="button"
                className={canRun ? "button primary" : "button secondary"}
                aria-disabled={!canRun}
                aria-describedby={requirement ? requirementId : undefined}
                onClick={() => {
                  if (canRun) {
                    try {
                      void task.start("/causal/sets/run", payload());
                    } catch (reason) {
                      describeError(reason);
                    }
                  }
                }}
              >
                <Play size={14} />
                {t("Run prompt set")}
              </button>
            </div>
            {requirement && (
              <p id={requirementId} className="action-requirement">
                {requirement}
              </p>
            )}
            {plan && (
              <div className="set-plan" role="status">
                <ErrorNotice message={plan.blockers} />
                <div className="set-facts">
                  <strong>
                    {t("{pairs} pairs, {layers} layers, {forwards} forwards", {
                      pairs: plan.pair_count,
                      layers: plan.layers.length,
                      forwards: plan.forward_passes,
                    })}
                  </strong>
                  <span>
                    {t("Prompt limit {count} tokens", {
                      count: plan.prompt_limit,
                    })}
                  </span>
                </div>
                <details>
                  <summary>{t("Resource estimates")}</summary>
                  <div className="set-facts">
                    <span>
                      {t("System memory")}{" "}
                      <bdi>{bytes(plan.estimated_host_bytes)}</bdi>
                    </span>
                    <span>
                      {t("GPU memory")}{" "}
                      <bdi>{bytes(plan.estimated_gpu_bytes)}</bdi>
                    </span>
                    <span>
                      {t("Saved evidence")}{" "}
                      <bdi>{bytes(plan.estimated_disk_bytes)}</bdi>
                    </span>
                  </div>
                  <p className="control-note">
                    {t(
                      "Estimates reserve 25 percent of available memory. The server checks resources again before running.",
                    )}
                  </p>
                </details>
              </div>
            )}
          </form>
        </Panel>
      )}
      {task.busy && (
        <div className="set-progress" role="status">
          <span>{t("Measuring prompt set")}</span>
          <progress
            aria-label={t("Prompt-set progress")}
            max={100}
            value={task.job?.progress || 0}
          />
          <button
            className="button secondary"
            disabled={!task.canCancel || task.cancelling}
            onClick={() => void task.cancel()}
          >
            <Square size={13} />
            {task.cancelling
              ? t("Stopping after the current forward pass")
              : t("Stop sweep")}
          </button>
        </div>
      )}
      {result && (
        <>
          {result.execution_id &&
            execution &&
            result.execution_id !== execution && (
              <div className="notice info">
                {t(
                  "These results belong to a previous model session. Their recorded model and settings are preserved.",
                )}
              </div>
            )}
          <PromptSetResults key={task.job!.id} result={result} />
          {task.job && (
            <div className="set-save">
              <SaveJobExperiment
                jobId={task.job.id}
                defaultName={t("Prompt-set patching")}
                disabled={task.busy}
              />
            </div>
          )}
        </>
      )}
    </>
  );
}
