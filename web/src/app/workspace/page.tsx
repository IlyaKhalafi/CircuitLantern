"use client";
import { useI18n } from "@/i18n/provider";
import {
  ArrowRight,
  ChevronRight,
  Layers3,
  Network,
  RefreshCw,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { CheckpointComparisonPanel } from "@/components/checkpoint-comparison";
import { ExecutionContext } from "@/components/execution-context";
import { MatrixResults } from "@/components/matrix-results";
import { SaveExperiment } from "@/components/save-experiment";
import { getLayer, ModelGraphView } from "@/components/model-graph";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  Metric,
  PageHeader,
  Panel,
} from "@/components/ui";
import { executionPath } from "@/lib/runtime";
import { api, number } from "@/lib/api";
import type { MatrixMetrics, ModelGraph, RuntimeState } from "@/lib/types";
export default function WorkspacePage() {
  const { t, rich } = useI18n();
  const [graph, setGraph] = useState<ModelGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [layer, setLayer] = useState<number | null>(null);
  const [selected, setSelected] = useState("");
  const [metrics, setMetrics] = useState<MatrixMetrics | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [view, setView] = useState("graph");
  const [comparisonBusy, setComparisonBusy] = useState(false);
  function chooseView(value: string) {
    setView(value);
    const url = new URL(window.location.href);
    if (value === "compare") url.searchParams.set("view", "compare");
    else url.searchParams.delete("view");
    window.history.replaceState(null, "", url);
  }
  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("view") === "compare")
      setView("compare");
  }, []);
  const requestId = useRef(0);
  const refreshId = useRef(0);
  const detail = useRef<HTMLDivElement>(null);
  const refresh = useCallback(async () => {
    const id = ++refreshId.current;
    ++requestId.current;
    setMetrics(null);
    setSelected("");
    setFilter("");
    setLayer(null);
    setGraph(null);
    setAnalyzing(false);
    setLoading(true);
    setError(null);
    try {
      const state = await api<RuntimeState>("/runtime");
      const next = state.loaded
        ? await api<ModelGraph>(
            executionPath("/graph", state.metadata?.execution_id),
          )
        : null;
      if (id === refreshId.current) {
        setGraph(next);
        setLayer(next && next.metadata.num_layers > 8 ? 0 : null);
      }
    } catch (reason) {
      if (id === refreshId.current) setError((reason as Error).message);
    } finally {
      if (id === refreshId.current) setLoading(false);
    }
  }, []);
  useEffect(() => {
    refresh();
  }, [refresh]);
  useEffect(() => {
    if (metrics)
      detail.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [metrics]);
  const inspect = useCallback(
    async (name: string) => {
      const id = ++requestId.current;
      setSelected(name);
      setMetrics(null);
      setAnalyzing(true);
      setError(null);
      try {
        const result = await api<MatrixMetrics>(
          executionPath(
            `/matrix?name=${encodeURIComponent(name)}`,
            graph?.metadata.execution_id,
          ),
        );
        if (id === requestId.current) {
          setMetrics(result);
        }
      } catch (e) {
        if (id === requestId.current) setError((e as Error).message);
      } finally {
        if (id === requestId.current) setAnalyzing(false);
      }
    },
    [graph?.metadata.execution_id],
  );
  const weights =
    graph?.matrices.filter(
      (w) =>
        (layer === null || getLayer(w.name) === layer) &&
        w.name.toLowerCase().includes(filter.toLowerCase()),
    ) || [];
  return (
    <>
      <PageHeader
        eyebrow={t("ANATOMY OF A TRANSFORMER")}
        title={t("Model explorer")}
        description={
          graph ? (
            <>
              <ExecutionContext metadata={graph.metadata} />
              <span className="revision-context">
                {t("Revision")} <bdi>{graph.metadata.revision.slice(0, 8)}</bdi>
              </span>
            </>
          ) : undefined
        }
        actions={
          <>
            <button
              className="button secondary"
              onClick={() =>
                chooseView(view === "compare" ? "graph" : "compare")
              }
            >
              {view === "compare"
                ? t("Back to explorer")
                : t("Compare checkpoints")}
            </button>
            <button
              className="button secondary"
              disabled={loading || comparisonBusy}
              onClick={refresh}
            >
              <RefreshCw size={14} />
              {t("Refresh explorer")}
            </button>
            {graph && (
              <Link href="/playground" className="button primary">
                {t("Trace a prompt")}
                <ArrowRight size={15} />
              </Link>
            )}
          </>
        }
      />
      <ErrorNotice message={error} />
      {view === "compare" ? (
        <CheckpointComparisonPanel
          graph={graph}
          onBusyChange={setComparisonBusy}
        />
      ) : loading ? (
        <Loading label={t("Opening model explorer")} />
      ) : !graph && error ? (
        <Panel
          title={t("Explorer temporarily unavailable")}
          subtitle={t("Retry after the runtime becomes available.")}
        >
          <div className="panel-body">
            <button className="button secondary" onClick={refresh}>
              {t("Retry explorer")}
            </button>
          </div>
        </Panel>
      ) : !graph ? (
        <EmptyState
          icon={<Network size={28} />}
          title={t("Your next model, under the microscope.")}
          href="/"
          action={t("Choose a model")}
        >
          {t(
            "Load a checkpoint from the library to explore its architecture, inspect weight matrices and discover their geometry.",
          )}
        </EmptyState>
      ) : (
        <>
          <div className="metrics-grid explorer-metrics">
            <Metric
              label={t("Parameters")}
              value={number(graph.total_parameters, 0)}
              help={t("Measured from loaded weights")}
            />
            <Metric
              label={t("Transformer blocks")}
              value={graph.metadata.num_layers}
              note={
                graph.dataflow?.some(
                  (flow) => flow.mixer_type === "linear_attention",
                )
                  ? t("{count} full attention blocks", {
                      count: graph.dataflow.filter(
                        (flow) => flow.mixer_type === "full_attention",
                      ).length,
                    })
                  : t("{v0} attention heads per block", {
                      v0: graph.metadata.num_heads,
                    })
              }
            />
            <Metric
              label={t("Hidden dimension")}
              value={graph.metadata.hidden_size}
              help={t("Width of each token's residual vector.")}
            />
            <Metric
              label={t("Weight matrices")}
              value={graph.matrices.length}
              help={t("Two dimensional parameters")}
            />
          </div>
          <div className="tabs" role="tablist" aria-label={t("Explorer view")}>
            <button
              role="tab"
              aria-selected={view === "graph"}
              className={view === "graph" ? "active" : ""}
              onClick={() => chooseView("graph")}
            >
              {t("Architecture graph")}
            </button>
            <button
              role="tab"
              aria-selected={view === "table"}
              className={view === "table" ? "active" : ""}
              onClick={() => chooseView("table")}
            >
              {t("Weight inventory")}
            </button>
          </div>
          {view === "graph" ? (
            <div className="graph-layout">
              <Panel
                title={t("Model architecture")}
                help={t(
                  "Scroll to zoom, drag to pan, select a block to inspect its matrices.",
                )}
                action={
                  <span className="pill subtle">
                    <Layers3 size={12} /> {graph.metadata.architecture}
                  </span>
                }
              >
                <ModelGraphView
                  graph={graph}
                  selectedLayer={layer}
                  onSelect={setLayer}
                  onInspect={inspect}
                />
              </Panel>
              <Panel
                title={
                  layer === null
                    ? t("All weight matrices")
                    : t("Block {v0} matrices", {
                        v0: layer,
                      })
                }
                subtitle={t("{v0} matrices in view", {
                  v0: weights.length,
                })}
                action={
                  layer !== null && (
                    <button
                      className="text-link icon-button"
                      onClick={() => setLayer(null)}
                    >
                      {t("Show all")}
                    </button>
                  )
                }
              >
                <div className="filter-input">
                  <input
                    dir="auto"
                    aria-label={t("Filter matrices")}
                    placeholder={t("Filter by matrix name\u2026")}
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                  />
                </div>
                <div className="weight-list">
                  {weights.map((w) => (
                    <button
                      key={w.name}
                      className={`weight-item ${selected === w.name ? "active" : ""}`}
                      onClick={() => inspect(w.name)}
                    >
                      <div>
                        <strong>{w.name}</strong>
                        <small>
                          {rich("{v0} \u00B7 {v1} elements", {
                            v0: w.shape.join(" × "),
                            v1: number(w.elements, 0),
                          })}
                        </small>
                      </div>
                      <ChevronRight size={15} />
                    </button>
                  ))}
                  {!weights.length && (
                    <div className="empty-inline">
                      {t("No matching matrices.")}
                    </div>
                  )}
                </div>
              </Panel>
            </div>
          ) : (
            <Panel
              title={t("Weight inventory")}
              subtitle={t("Select any matrix to compute its properties.")}
            >
              <div className="filter-input">
                <input
                  dir="auto"
                  aria-label={t("Filter matrices")}
                  placeholder={t("Filter by matrix name\u2026")}
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                />
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>{t("MATRIX")}</th>
                      <th>{t("SHAPE")}</th>
                      <th>{t("ELEMENTS")}</th>
                      <th>{t("DTYPE")}</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {weights.map((w) => (
                      <tr
                        key={w.name}
                        className="selectable"
                        onClick={() => inspect(w.name)}
                        tabIndex={0}
                        onKeyDown={(e) => e.key === "Enter" && inspect(w.name)}
                      >
                        <td className="code">{w.name}</td>
                        <td className="code">{w.shape.join(" × ")}</td>
                        <td>{t(number(w.elements, 0))}</td>
                        <td>{w.dtype}</td>
                        <td>
                          <ChevronRight size={14} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
          <div ref={detail} className="section-gap matrix-detail">
            {analyzing && (
              <Loading
                label={t("Computing properties of {v0}", {
                  v0: selected,
                })}
              />
            )}
            {metrics && (
              <MatrixResults
                metrics={metrics}
                model={graph.metadata}
                footer={
                  metrics.analysis_id && (
                    <SaveExperiment
                      key={metrics.analysis_id}
                      source={{
                        kind: "matrix",
                        id: metrics.analysis_id,
                        executionId: metrics.execution_id ?? undefined,
                      }}
                      defaultName={metrics.name}
                    />
                  )
                }
              />
            )}
          </div>
        </>
      )}
    </>
  );
}
