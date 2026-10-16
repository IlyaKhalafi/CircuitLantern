"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ArrowRight, Download, GitCompareArrows, Trash2 } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Loading, Metric, Panel } from "@/components/ui";
import { SeriesChart } from "@/components/charts";
import { api, downloadJSON, number, post } from "@/lib/api";
import type { ModelGraph } from "@/lib/types";
import type {
  CheckpointComparison,
  CheckpointReference,
  CheckpointComparisonSummary,
  CheckpointLibrary,
} from "@/lib/checkpoints";

export function CheckpointComparisonPanel({
  graph,
  onBusyChange,
}: {
  graph: ModelGraph | null;
  onBusyChange: (busy: boolean) => void;
}) {
  const { t, dateTime } = useI18n();
  const [references, setReferences] = useState<CheckpointReference[]>([]);
  const [referenceId, setReferenceId] = useState("");
  const [history, setHistory] = useState<CheckpointComparisonSummary[]>([]);
  const [result, setResult] = useState<CheckpointComparison | null>(null);
  const [name, setName] = useState("Base reference");
  const [prompt, setPrompt] = useState("The capital of France is");
  const [names, setNames] = useState<string[]>([]);
  const [filter, setFilter] = useState("");
  const [busy, setBusy] = useState(false);
  const [setupOpen, setSetupOpen] = useState(true);
  const [reading, setReading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmReportDelete, setConfirmReportDelete] = useState(false);
  const [unreadable, setUnreadable] = useState<{
    reference: string[];
    comparison: string[];
  }>({ reference: [], comparison: [] });
  const sequence = useRef(0);
  const reference = references.find((item) => item.id === referenceId);
  useEffect(() => {
    setConfirmReportDelete(false);
    if (result) setSetupOpen(false);
  }, [result?.id]);
  useEffect(() => {
    let active = true;
    Promise.all([
      api<CheckpointLibrary<CheckpointReference>>("/checkpoints/references"),
      api<CheckpointLibrary<CheckpointComparisonSummary>>(
        "/checkpoints/comparisons",
      ),
    ])
      .then(([refs, reports]) => {
        if (!active) return;
        setReferences(refs.items);
        setHistory(reports.items);
        setUnreadable({
          reference: refs.unreadable,
          comparison: reports.unreadable,
        });
        const saved = localStorage.getItem("lantern-checkpoint-reference");
        setReferenceId(
          refs.items.some((item) => item.id === saved)
            ? saved!
            : refs.items[0]?.id || "",
        );
      })
      .catch((reason) => {
        if (active) setError(reason.message);
      })
      .finally(() => {
        if (active) setReading(false);
      });
    return () => {
      active = false;
      ++sequence.current;
    };
  }, []);
  useEffect(() => {
    const preferred = graph?.matrices.find((item) =>
      /(?:q_proj|c_attn|query_key_value)\.weight$/.test(item.name),
    );
    setNames(
      preferred
        ? [preferred.name]
        : graph?.matrices.slice(0, 1).map((item) => item.name) || [],
    );
  }, [graph?.metadata.execution_id]);
  useEffect(() => {
    onBusyChange(busy);
    return () => onBusyChange(false);
  }, [busy, onBusyChange]);
  function selectReference(id: string) {
    ++sequence.current;
    setReferenceId(id);
    setResult(null);
    setError(null);
    setBusy(false);
    setConfirmDelete(false);
    localStorage.setItem("lantern-checkpoint-reference", id);
  }
  async function capture(event: React.FormEvent) {
    event.preventDefault();
    const id = ++sequence.current;
    setBusy(true);
    setError(null);
    try {
      const captured = await post<CheckpointReference>(
        "/checkpoints/references",
        {
          name: name.trim(),
          prompt,
          matrices: names,
          expected_execution_id: graph?.metadata.execution_id,
        },
      );
      if (id !== sequence.current) return;
      setReferences((items) => [captured, ...items]);
      selectReference(captured.id);
    } catch (reason) {
      if (id === sequence.current) setError((reason as Error).message);
    } finally {
      if (id === sequence.current) setBusy(false);
    }
  }
  async function compare() {
    const id = ++sequence.current;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const report = await post<CheckpointComparison>("/checkpoints/compare", {
        reference_id: referenceId,
        expected_execution_id: graph?.metadata.execution_id,
      });
      if (id !== sequence.current) return;
      setResult(report);
      setHistory((items) => [report, ...items]);
    } catch (reason) {
      if (id === sequence.current) setError((reason as Error).message);
    } finally {
      if (id === sequence.current) setBusy(false);
    }
  }
  async function remove() {
    const id = ++sequence.current;
    setBusy(true);
    setError(null);
    try {
      await api("/checkpoints/references/" + referenceId, { method: "DELETE" });
      if (id !== sequence.current) return;
      setReferences((items) => items.filter((item) => item.id !== referenceId));
      selectReference("");
    } catch (reason) {
      if (id === sequence.current) setError((reason as Error).message);
    } finally {
      if (id === sequence.current) setBusy(false);
    }
  }
  async function openReport(identifier: string) {
    const id = ++sequence.current;
    setBusy(true);
    setError(null);
    setConfirmReportDelete(false);
    if (!identifier) {
      setResult(null);
      setBusy(false);
      return;
    }
    try {
      const value = await api<CheckpointComparison>(
        "/checkpoints/comparisons/" + identifier,
      );
      if (id === sequence.current) setResult(value);
    } catch (reason) {
      if (id === sequence.current) setError((reason as Error).message);
    } finally {
      if (id === sequence.current) setBusy(false);
    }
  }
  async function deleteReport() {
    if (!result) return;
    const id = ++sequence.current;
    setBusy(true);
    setError(null);
    try {
      await api("/checkpoints/comparisons/" + result.id, { method: "DELETE" });
      if (id !== sequence.current) return;
      setHistory((items) => items.filter((item) => item.id !== result.id));
      setResult(null);
      setConfirmReportDelete(false);
    } catch (reason) {
      if (id === sequence.current) setError((reason as Error).message);
    } finally {
      if (id === sequence.current) setBusy(false);
    }
  }
  async function removeUnreadable(
    kind: "reference" | "comparison",
    identifier: string,
  ) {
    setBusy(true);
    setError(null);
    try {
      await api(
        "/checkpoints/" +
          (kind === "reference" ? "references/" : "comparisons/") +
          identifier,
        { method: "DELETE" },
      );
      setUnreadable((items) => ({
        ...items,
        [kind]: items[kind].filter((item) => item !== identifier),
      }));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const captureForm = (
    <form className="checkpoint-form" onSubmit={capture}>
      <p className="inline-note">
        {t(
          "Capture the loaded base model first. Reference evidence stays available after loading a finetuned model or restarting the server.",
        )}
      </p>
      <div className="form-grid">
        <label>
          {t("Reference name")}
          <input
            dir="auto"
            className="form-input"
            aria-label={t("Reference name")}
            value={name}
            maxLength={100}
            disabled={busy}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <label>
          {t("Primary matrix")}
          <select
            aria-label={t("Primary matrix")}
            value={names[0] || ""}
            disabled={busy}
            onChange={(e) => setNames([e.target.value])}
          >
            {graph?.matrices.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name} · {item.shape.join(" × ")}
              </option>
            ))}
          </select>
        </label>
      </div>
      <label>
        {t("Comparison prompt")}
        <textarea
          dir="auto"
          aria-label={t("Comparison prompt")}
          value={prompt}
          maxLength={8000}
          disabled={busy}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </label>
      <details className="checkpoint-extra">
        <summary>{t("Compare additional matrices")}</summary>
        <p className="inline-note">
          {t(
            "Choose up to 12 matrices. Each reference is limited to 128 MiB of actual weights and activation evidence.",
          )}
        </p>
        <input
          dir="auto"
          className="form-input"
          aria-label={t("Filter comparison matrices")}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={t("Filter by matrix name…")}
        />
        <div className="checkpoint-matrix-options">
          {graph?.matrices
            .filter((item) =>
              item.name.toLowerCase().includes(filter.toLowerCase()),
            )
            .map((item) => (
              <label key={item.name}>
                <input
                  type="checkbox"
                  disabled={
                    busy || (!names.includes(item.name) && names.length >= 12)
                  }
                  checked={names.includes(item.name)}
                  onChange={(e) =>
                    setNames((items) =>
                      e.target.checked
                        ? [...items, item.name]
                        : items.filter((value) => value !== item.name),
                    )
                  }
                />
                <span>
                  {item.name}
                  <small>{item.shape.join(" × ")}</small>
                </span>
              </label>
            ))}
        </div>
      </details>
      <button
        className="button primary"
        type="submit"
        disabled={
          busy || !graph || !name.trim() || !prompt.trim() || !names.length
        }
      >
        {t("Capture reference")}
        <ArrowRight size={14} />
      </button>
    </form>
  );
  return (
    <div className="checkpoint-workflow">
      <details
        className="checkpoint-setup"
        open={setupOpen}
        onToggle={(event) => setSetupOpen(event.currentTarget.open)}
      >
        <summary>
          {result
            ? t("Review comparison setup")
            : t("Set up a checkpoint comparison")}
        </summary>
        <Panel
          title={t("Compare base and finetuned checkpoints")}
          subtitle={t(
            "One model in memory. A retained reference. Measurements of what changed.",
          )}
        >
          <ol className="checkpoint-steps">
            {[
              "Capture a reference",
              "Load the other checkpoint",
              "Measure the changes",
            ].map((label, index) => (
              <li key={label}>
                <span>{index + 1}</span>
                {t(label)}
              </li>
            ))}
          </ol>
          {reading ? (
            <Loading />
          ) : references.length ? (
            <div className="checkpoint-form">
              <label>
                {t("Reference capture")}
                <select
                  aria-label={t("Reference capture")}
                  disabled={busy}
                  value={referenceId}
                  onChange={(e) => selectReference(e.target.value)}
                >
                  <option value="">{t("Choose a reference capture")}</option>
                  {references.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name} · {item.model_id} ·{" "}
                      {item.revision.slice(0, 8)}
                    </option>
                  ))}
                </select>
              </label>
              {reference && (
                <div className="checkpoint-reference">
                  <strong>
                    {reference.model_id} · {reference.revision.slice(0, 12)}
                  </strong>
                  <p dir="auto">{reference.prompt}</p>
                  <small>
                    {t("{count} matrices retained", {
                      count: reference.matrices.length,
                    })}{" "}
                    · {dateTime(reference.created_at)}
                  </small>
                </div>
              )}
              <div className="prompt-toolbar">
                <Link href="/?return=compare" className="button secondary">
                  {t("Load another checkpoint")}
                </Link>
                <button
                  className="button primary"
                  disabled={busy || !graph || !reference}
                  onClick={compare}
                >
                  <GitCompareArrows size={15} />
                  {t("Compare with loaded model")}
                </button>
              </div>
              {graph ? (
                <p className="inline-note">
                  {t("Loaded checkpoint")} · {graph.metadata.id} ·{" "}
                  {graph.metadata.revision.slice(0, 12)}
                </p>
              ) : (
                <p className="inline-note">
                  {t(
                    "Load a model to create a comparison. Saved reports remain available below.",
                  )}
                </p>
              )}
              {reference && (
                <details className="checkpoint-extra">
                  <summary>{t("Manage reference capture")}</summary>
                  <p className="inline-note">
                    {t(
                      "Deleting a reference frees its tensor storage. Saved comparison reports keep their measurements and provenance.",
                    )}
                  </p>
                  <button
                    className="button danger"
                    disabled={busy}
                    onClick={() =>
                      confirmDelete ? void remove() : setConfirmDelete(true)
                    }
                  >
                    <Trash2 size={14} />
                    {confirmDelete
                      ? t("Confirm delete reference")
                      : t("Delete reference")}
                  </button>
                  {confirmDelete && (
                    <button
                      className="button secondary"
                      onClick={() => setConfirmDelete(false)}
                    >
                      {t("Cancel")}
                    </button>
                  )}
                </details>
              )}
              {graph && (
                <details className="checkpoint-extra">
                  <summary>{t("Capture another reference")}</summary>
                  {captureForm}
                </details>
              )}
            </div>
          ) : graph ? (
            captureForm
          ) : (
            <div className="empty-inline">
              <Link className="button primary" href="/?return=compare">
                {t("Load a base checkpoint")}
              </Link>
            </div>
          )}
          <ErrorNotice message={error} />
          {busy && <Loading label={t("Measuring checkpoint evidence")} />}
        </Panel>
      </details>
      {!!history.length && (
        <Panel
          title={t("Saved checkpoint comparisons")}
          subtitle={t("Reports remain available after unloading either model.")}
        >
          <div className="checkpoint-form">
            <label>
              {t("Open a comparison report")}
              <select
                aria-label={t("Open a comparison report")}
                value={result?.id || ""}
                disabled={busy}
                onChange={(event) => void openReport(event.target.value)}
              >
                <option value="">{t("Choose a saved report")}</option>
                {history.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.reference.model_id} → {item.selected.model_id} ·{" "}
                    {dateTime(item.created_at)}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </Panel>
      )}
      {(unreadable.reference.length > 0 ||
        unreadable.comparison.length > 0) && (
        <details className="checkpoint-extra report-details">
          <summary>{t("Unreadable saved evidence")}</summary>
          <p>
            {t(
              "Interrupted or invalid evidence files are isolated from valid captures. Remove an unreadable file to reclaim its storage.",
            )}
          </p>
          {(["reference", "comparison"] as const).flatMap((kind) =>
            unreadable[kind].map((id) => (
              <div key={id}>
                <code>{id}</code>
                <button
                  className="button danger"
                  disabled={busy}
                  onClick={() => void removeUnreadable(kind, id)}
                >
                  {t("Remove unreadable evidence")}
                </button>
              </div>
            )),
          )}
        </details>
      )}
      {result && (
        <>
          <CheckpointReport result={result} />
          <details className="checkpoint-extra report-details">
            <summary>{t("Manage comparison report")}</summary>
            <button
              className="button danger"
              disabled={busy}
              onClick={() =>
                confirmReportDelete
                  ? void deleteReport()
                  : setConfirmReportDelete(true)
              }
            >
              {confirmReportDelete
                ? t("Confirm delete report")
                : t("Delete report")}
            </button>
            {confirmReportDelete && (
              <button
                className="button secondary"
                onClick={() => setConfirmReportDelete(false)}
              >
                {t("Cancel")}
              </button>
            )}
          </details>
        </>
      )}
    </div>
  );
}

function CheckpointReport({ result }: { result: CheckpointComparison }) {
  const { t } = useI18n();
  const maximum = result.layers
    .filter((item) => item.relative_distance !== null)
    .sort((a, b) => b.relative_distance! - a.relative_distance!)[0];
  return (
    <div className="checkpoint-report">
      <div className="section-heading">
        <div>
          <h2>{t("Checkpoint changes")}</h2>
          <p>
            {result.reference.model_id} → {result.selected.model_id}
          </p>
        </div>
        <button
          className="button secondary"
          onClick={() =>
            downloadJSON(
              result,
              "checkpoint-comparison-" + result.id.slice(0, 8) + ".json",
            )
          }
        >
          <Download size={14} />
          {t("Export comparison")}
        </button>
      </div>
      {result.warnings.map((message) => (
        <div className="notice info" key={message}>
          {t(message)}
        </div>
      ))}
      <div className="metrics-grid">
        <Metric
          label={t("Prediction KL")}
          value={number(result.kl_reference_to_selected, 5)}
          note={t("Reference to selected · nats")}
        />
        <Metric
          label={t("Total variation")}
          value={(result.total_variation * 100).toFixed(2) + "%"}
          note={t("Full next-token distribution")}
        />
        <Metric
          label={t("Largest residual drift")}
          value={
            maximum ? t("Layer {v0}", { v0: maximum.layer }) : t("Unavailable")
          }
          note={t("Final prompt token · raw block output")}
        />
        <Metric
          label={t("Matrices compared")}
          value={result.weights.length}
          note={t("Actual aligned weight distances")}
        />
      </div>
      <Panel
        title={t("Residual drift by layer")}
        subtitle={t(
          "Actual residual vector distance divided by the reference vector norm. The same prompt inputs are used in both checkpoints.",
        )}
      >
        <SeriesChart
          data={result.layers.map((item) => ({
            index: item.layer,
            drift:
              item.relative_distance === null
                ? null
                : item.relative_distance * 100,
          }))}
          series={[
            {
              key: "drift",
              label: "Relative residual drift (%)",
              color: "var(--accent)",
            },
          ]}
        />
      </Panel>
      <Panel
        title={t("Weight changes")}
        subtitle={t(
          "Float32 logical matrices, with quantized weights dequantized before comparison.",
        )}
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t("MATRIX")}</th>
                <th>{t("RELATIVE CHANGE")}</th>
                <th>{t("COSINE SIMILARITY")}</th>
                <th>{t("RMS CHANGE")}</th>
              </tr>
            </thead>
            <tbody>
              {result.weights.map((item) => (
                <tr key={item.name}>
                  <th scope="row" className="code">
                    {item.name}
                  </th>
                  <td>
                    {item.relative_distance === null
                      ? t("Undefined")
                      : (item.relative_distance * 100).toFixed(3) + "%"}
                  </td>
                  <td>
                    {item.cosine_similarity === null
                      ? t("Undefined")
                      : number(item.cosine_similarity, 6)}
                  </td>
                  <td>{number(item.rms_change, 6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <details className="checkpoint-extra report-details">
        <summary>{t("Inspect predictions and layer measurements")}</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t("LAYER")}</th>
                <th>{t("VECTOR DISTANCE")}</th>
                <th>{t("COSINE SIMILARITY")}</th>
                <th>{t("REFERENCE PREDICTION")}</th>
                <th>{t("SELECTED PREDICTION")}</th>
              </tr>
            </thead>
            <tbody>
              {result.layers.map((item) => (
                <tr key={item.layer}>
                  <th scope="row">{item.layer}</th>
                  <td>{number(item.distance, 5)}</td>
                  <td>
                    {item.cosine_similarity === null
                      ? t("Undefined")
                      : number(item.cosine_similarity, 5)}
                  </td>
                  <td className="code">{item.reference_prediction.token}</td>
                  <td className="code">{item.selected_prediction.token}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="inline-note">
          {t(
            "Intermediate predictions use a diagnostic logit lens. They are not calibrated probabilities.",
          )}
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t("TOKEN")}</th>
                <th>{t("REFERENCE PROBABILITY")}</th>
                <th>{t("SELECTED PROBABILITY")}</th>
                <th>{t("LOG PROBABILITY CHANGE")}</th>
              </tr>
            </thead>
            <tbody>
              {[...result.predictions]
                .sort(
                  (a, b) =>
                    Math.abs(b.selected_probability - b.reference_probability) -
                    Math.abs(a.selected_probability - a.reference_probability),
                )
                .map((item) => (
                  <tr key={item.token_id}>
                    <th scope="row" className="code">
                      {item.token}
                    </th>
                    <td>{(item.reference_probability * 100).toFixed(3)}%</td>
                    <td>{(item.selected_probability * 100).toFixed(3)}%</td>
                    <td>{number(item.log_probability_change, 5)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </details>
      <details className="checkpoint-extra report-details">
        <summary>{t("Comparison provenance and conventions")}</summary>
        <p>
          {t(
            "Weight and residual distances assume matching coordinates across checkpoints. They measure changes and do not establish causal effects.",
          )}
        </p>
        <dl>
          <dt>{t("Reference checkpoint")}</dt>
          <dd>
            {result.reference.model_id} · {result.reference.revision}
          </dd>
          <dt>{t("Selected checkpoint")}</dt>
          <dd>
            {result.selected.model_id} · {result.selected.revision}
          </dd>
        </dl>
        <p dir="auto">{result.reference.prompt}</p>
      </details>
    </div>
  );
}
