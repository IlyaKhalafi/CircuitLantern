"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowUpRight, Download } from "lucide-react";
import { Heatmap, SeriesChart } from "@/components/charts";
import { InductionResults } from "@/components/induction-results";
import { SavedReferencePicker } from "@/components/saved-reference-picker";
import { ErrorNotice, Loading, Metric, Panel } from "@/components/ui";
import { useProbeComparison } from "@/hooks/use-probe-comparison";
import { useExperiment } from "@/lib/use-experiment";
import {
  matchingProbeHref,
  type ProbeComparison,
  type ProbeComparisonHead,
} from "@/lib/induction-comparison";
import { downloadJSON, number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import { useI18n } from "@/i18n/provider";
import type { Experiment } from "@/lib/types";
import type { InductionResult } from "@/lib/induction";
import "./saved-induction-comparison.css";

const metrics = {
  prefix: {
    label: "Prefix matching",
    selected: "selected_prefix",
    reference: "reference_prefix",
    change: "prefix_difference",
    paired: "pattern_paired_examples",
    factor: 100,
  },
  duplicate: {
    label: "Duplicate token",
    selected: "selected_duplicate",
    reference: "reference_duplicate",
    change: "duplicate_difference",
    paired: "pattern_paired_examples",
    factor: 100,
  },
  previous: {
    label: "Previous token",
    selected: "selected_previous",
    reference: "reference_previous",
    change: "previous_difference",
    paired: "pattern_paired_examples",
    factor: 100,
  },
  causal: {
    label: "Delta log probability",
    selected: "selected_causal_delta",
    reference: "reference_causal_delta",
    change: "causal_delta_difference",
    paired: "causal_paired_examples",
    factor: 1,
  },
  kl: {
    label: "KL divergence",
    selected: "selected_kl",
    reference: "reference_kl",
    change: "kl_difference",
    paired: "causal_paired_examples",
    factor: 1,
  },
} as const;

function Source({
  label,
  record,
  result,
}: {
  label: string;
  record: Experiment;
  result: InductionResult;
}) {
  const { t } = useI18n();
  return (
    <div className="probe-comparison-source">
      <span className="eyebrow">{t(label)}</span>
      <h3 dir="auto">{record.name}</h3>
      <p>
        <bdi>{result.model_id}</bdi>
      </p>
      <p>{t("revision {v0}", { v0: result.revision.slice(0, 12) })}</p>
      <p>
        {t(executionLabel(result.execution_precision, result.execution_config))}{" "}
        · {result.device.toUpperCase()}
      </p>
      <span className={"pill " + (result.complete ? "green" : "orange")}>
        {result.complete ? t("Complete") : t("Partial")}
      </span>
      <Link href={"/experiments?experiment=" + record.id}>
        {t("Open captured evidence")}
        <ArrowUpRight size={12} />
      </Link>
    </div>
  );
}

function Changes({ comparison }: { comparison: ProbeComparison }) {
  const { t } = useI18n();
  const [metric, setMetric] = useState<keyof typeof metrics>("prefix");
  const [layer, setLayer] = useState("all");
  const [pairedOnly, setPairedOnly] = useState(true);
  const [page, setPage] = useState(0);
  const definition = metrics[metric];
  const layers = [...new Set(comparison.heads.map((row) => row.layer))].sort(
    (a, b) => a - b,
  );
  const magnitude = (row: ProbeComparisonHead) =>
    row[definition.change] === null ? -1 : Math.abs(row[definition.change]!);
  const rows = comparison.heads
    .filter(
      (row) =>
        (layer === "all" || row.layer === +layer) &&
        (!pairedOnly || row[definition.paired] > 0),
    )
    .sort(
      (a, b) =>
        magnitude(b) - magnitude(a) || a.layer - b.layer || a.head - b.head,
    );
  const pages = Math.max(1, Math.ceil(rows.length / 12));
  const current = Math.min(page, pages - 1);
  const columns = Math.max(1, ...comparison.heads.map((row) => row.head + 1));
  const indexed = new Map(
    comparison.heads.map((row) => [
      row.layer + "/" + row.head,
      row[definition.change],
    ]),
  );
  const visibleLayers = layers.filter(
    (item) =>
      (layer === "all" || item === +layer) &&
      (!pairedOnly ||
        comparison.heads.some(
          (row) => row.layer === item && row[definition.paired] > 0,
        )),
  );
  const pairedHeads = comparison.heads.filter(
    (row) => row[definition.paired] > 0,
  ).length;
  const heatmap = visibleLayers.map((item) =>
    Array.from({ length: columns }, (_, head) => {
      const value = indexed.get(item + "/" + head);
      return value == null ? null : value * definition.factor;
    }),
  );
  const display = (value: number | null, signed = false) =>
    value === null
      ? t("Unavailable")
      : (signed && value > 0 ? "+" : "") + number(value * definition.factor, 5);
  return (
    <>
      <div className="metrics-grid probe-comparison-metrics">
        <Metric
          label={t("Shared probe examples")}
          value={comparison.shared_examples}
        />
        <Metric
          label={t("Paired attention heads")}
          value={comparison.shared_pattern_heads}
        />
        <Metric
          label={t("Paired causal heads")}
          value={comparison.shared_causal_heads}
        />
        <Metric
          label={t("Repeated loss change")}
          value={number(comparison.repeated_loss_difference, 5)}
          note={t("Selected minus reference in nats")}
        />
      </div>
      <div className="probe-comparison-controls">
        <label>
          {t("Metric to compare")}
          <select
            aria-label={t("Probe comparison metric")}
            value={metric}
            onChange={(event) => {
              setMetric(event.target.value as keyof typeof metrics);
              setPage(0);
            }}
          >
            {Object.entries(metrics).map(([key, value]) => (
              <option value={key} key={key}>
                {t(value.label)}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t("Layer")}
          <select
            aria-label={t("Probe comparison layer")}
            value={layer}
            onChange={(event) => {
              setLayer(event.target.value);
              setPage(0);
            }}
          >
            <option value="all">{t("All tested layers")}</option>
            {layers.map((item) => (
              <option value={item} key={item}>
                {t("Layer {v0}", { v0: item })}
              </option>
            ))}
          </select>
        </label>
        <label className="probe-paired-only">
          <input
            type="checkbox"
            checked={pairedOnly}
            onChange={(event) => {
              setPairedOnly(event.target.checked);
              setPage(0);
            }}
          />
          {t("Paired heads only")}
        </label>
      </div>
      <p className="probe-comparison-note">
        {definition.factor === 100
          ? t(
              "Attention changes are percentage points. Selected and reference values are percentages.",
            )
          : t(
              "Causal changes are differences between the two within-checkpoint ablation effects, in nats.",
            )}
      </p>
      <div className="two-column probe-comparison-charts">
        <Panel
          title={t("Loss across checkpoints")}
          subtitle={t(
            "Mean negative log probability on the same completed probe examples.",
          )}
        >
          <SeriesChart
            data={comparison.loss_by_repeat.map((row) => ({
              index: row.repeat,
              selected: row.selected,
              reference: row.reference,
            }))}
            series={[
              { key: "selected", label: "Selected", color: "var(--accent)" },
              { key: "reference", label: "Reference", color: "var(--blue)" },
            ]}
            interpolation="linear"
            showPoints
            height={220}
          />
        </Panel>
        <Panel
          title={t("Head changes")}
          subtitle={t(
            "Measured changes for {v0} of {v1} head coordinates. Blank cells are unpaired measurements.",
            { v0: pairedHeads, v1: comparison.heads.length },
          )}
        >
          {!visibleLayers.length ? (
            <p className="probe-comparison-note">
              {t("No paired head measurements for this metric.")}
            </p>
          ) : (
            <Heatmap
              data={heatmap}
              height={Math.min(280, Math.max(100, visibleLayers.length * 16))}
              label={t("Probe head differences by layer and query head")}
              xLabels={Array.from({ length: columns }, (_, head) =>
                t("Head {v0}", { v0: head }),
              )}
              yLabels={visibleLayers.map((item) =>
                t("Layer {v0}", { v0: item }),
              )}
            />
          )}
          <p className="probe-comparison-note">
            {t(
              "Blue indicates a decrease. Gold indicates an increase. Color intensity is relative to the largest absolute change.",
            )}
          </p>
        </Panel>
      </div>
      <p className="probe-table-hint">
        {t("Scroll horizontally to see all columns.")}
      </p>
      <div
        className="table-wrap probe-comparison-table"
        role="region"
        aria-label={t("Paired probe head changes")}
        tabIndex={0}
      >
        <table>
          <caption className="sr-only">
            {t("Paired probe head changes")}
          </caption>
          <thead>
            <tr>
              {[
                ["Layer", t("Layer")],
                ["Head", t("Head")],
                ["Selected", t("Selected")],
                ["Reference", t("Reference")],
                ["Change", t("Change")],
                ["Paired probes", t("Paired probes")],
              ].map(([key, label]) => (
                <th scope="col" key={key}>
                  {label}
                  {["Selected", "Reference", "Change"].includes(key) && (
                    <small className="probe-comparison-unit">
                      {definition.factor === 100
                        ? key === "Change"
                          ? t("percentage points")
                          : "%"
                        : t("nats")}
                    </small>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows
              .slice(current * 12, (current + 1) * 12)
              .map((row: ProbeComparisonHead) => (
                <tr key={row.layer + "/" + row.head}>
                  <td>{row.layer}</td>
                  <td>{row.head}</td>
                  <td>
                    <bdi>{display(row[definition.selected])}</bdi>
                  </td>
                  <td>
                    <bdi>{display(row[definition.reference])}</bdi>
                  </td>
                  <td
                    className={
                      row[definition.change] == null
                        ? ""
                        : row[definition.change]! < 0
                          ? "probe-negative"
                          : "probe-positive"
                    }
                  >
                    <bdi>{display(row[definition.change], true)}</bdi>
                  </td>
                  <td>{row[definition.paired]}</td>
                </tr>
              ))}
          </tbody>
        </table>
        {!rows.length && (
          <p className="probe-comparison-note">
            {t("No paired head measurements for this metric.")}
          </p>
        )}
      </div>
      <div className="probe-comparison-pagination">
        <button
          type="button"
          className="button secondary"
          disabled={current === 0}
          onClick={() => setPage(current - 1)}
        >
          {t("Previous")}
        </button>
        <span aria-live="polite" aria-atomic="true">
          {t("Page {v0} of {v1}", { v0: current + 1, v1: pages })}
        </span>
        <button
          type="button"
          className="button secondary"
          disabled={current + 1 >= pages}
          onClick={() => setPage(current + 1)}
        >
          {t("Next")}
        </button>
      </div>
      <p className="probe-comparison-note">
        {t(
          "Ranked changes are exploratory. Heads were selected and measured on these same probes. No uncertainty interval is estimated.",
        )}
      </p>
      <p className="probe-comparison-note">
        {t(
          "Ordered by absolute change. Head indices align coordinates and do not establish the same learned function. These synthetic probes do not measure general model quality.",
        )}
      </p>
    </>
  );
}

export function SavedInductionComparison({
  experiment,
  result,
  experiments,
  libraryLoading,
  libraryError,
  onRetryLibrary,
}: {
  experiment: Experiment;
  result: InductionResult;
  experiments: Experiment[];
  libraryLoading: boolean;
  libraryError: string | null;
  onRetryLibrary: () => void;
}) {
  const { t } = useI18n();
  const router = useRouter();
  const params = useSearchParams();
  const referenceId = params.get("compare");
  const reference = useExperiment(referenceId);
  const compared = useProbeComparison(experiment.id, referenceId);
  const [actionError, setActionError] = useState<string | null>(null);
  const referenceResult =
    reference.selected?.kind === "induction_probe"
      ? reference.selected.result
      : null;
  const candidates = experiments.filter(
    (item) => item.id !== experiment.id && item.kind === "induction_probe",
  );
  const error =
    reference.error ||
    compared.error ||
    (referenceId === experiment.id
      ? t("Choose a different saved probe as the reference.")
      : null);
  const ready = compared.data && referenceResult && !error;
  return (
    <>
      <details
        className="capture-details probe-comparison"
        open={!!referenceId}
      >
        <summary>{t("Compare probe checkpoints")}</summary>
        <Panel
          title={t("Compare saved probe evidence")}
          subtitle={t(
            "Compare base and finetuned checkpoints using aligned inputs, token mappings and execution settings.",
          )}
        >
          <SavedReferencePicker
            label={t("Reference induction probe")}
            referenceId={referenceId}
            reference={reference.selected}
            candidates={candidates}
            libraryLoading={libraryLoading}
            libraryError={libraryError}
            onRetryLibrary={onRetryLibrary}
            emptyText="Save another matching probe to compare checkpoint changes here."
          >
            {result.format === "circuit-lantern-induction-probe-v2" && (
              <button
                type="button"
                className="button secondary"
                onClick={() => {
                  try {
                    router.push(matchingProbeHref(experiment.id));
                  } catch (reason) {
                    setActionError((reason as Error).message);
                  }
                }}
              >
                {t("Repeat with another checkpoint")}
                <ArrowUpRight size={14} />
              </button>
            )}
            {ready && (
              <button
                type="button"
                className="button secondary"
                onClick={() =>
                  downloadJSON(
                    {
                      ...compared.data,
                      selected: experiment,
                      reference: reference.selected,
                    },
                    "induction-comparison-" +
                      experiment.id.slice(0, 8) +
                      "-" +
                      referenceId!.slice(0, 8) +
                      ".json",
                  )
                }
              >
                <Download size={14} />
                {t("Export comparison")}
              </button>
            )}
          </SavedReferencePicker>
          {(compared.loading || reference.loading) && (
            <Loading label={t("Comparing saved probe measurements")} />
          )}
          <ErrorNotice message={error || actionError} />
          {compared.error && (
            <button
              type="button"
              className="button secondary probe-retry"
              onClick={compared.retry}
            >
              {t("Retry comparison")}
            </button>
          )}
          {reference.selected && referenceResult && (
            <div className="probe-comparison-sources">
              <Source label="Selected" record={experiment} result={result} />
              <Source
                label="Reference"
                record={reference.selected}
                result={referenceResult}
              />
            </div>
          )}
          {ready && (
            <div className="probe-comparison-body">
              <p className="notice info">
                {t(
                  "{v0} shared completed probes. Attention and loss are recomputed on that intersection. Each causal head uses its own shared completed pairs.",
                  { v0: compared.data!.shared_examples },
                )}
              </p>
              <Changes
                key={experiment.id + "/" + referenceId}
                comparison={compared.data!}
              />
            </div>
          )}
        </Panel>
      </details>
      <InductionResults result={result} />
    </>
  );
}
