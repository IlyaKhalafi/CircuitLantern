"use client";

import {
  ArrowRight,
  ChevronRight,
  Download,
  Layers3,
  Network,
  Search,
  RefreshCw,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { MatrixResults } from "@/components/matrix-results";
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
import { api, downloadJSON, number } from "@/lib/api";
import type { MatrixMetrics, ModelGraph, RuntimeState } from "@/lib/types";

export default function WorkspacePage() {
  const [graph, setGraph] = useState<ModelGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [layer, setLayer] = useState<number | null>(null);
  const [selected, setSelected] = useState("");
  const [metrics, setMetrics] = useState<MatrixMetrics | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [view, setView] = useState("graph");
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
      if (id === refreshId.current) setGraph(next);
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
        eyebrow="ANATOMY OF A TRANSFORMER"
        title="See the structure. Inspect the weights."
        description={
          graph
            ? `${graph.metadata.id} · ${graph.metadata.architecture} · revision ${graph.metadata.revision.slice(0, 8)}`
            : "A map of the computations behind every prediction."
        }
        actions={
          <>
            <button
              className="button secondary"
              disabled={loading}
              onClick={refresh}
            >
              <RefreshCw size={14} /> Refresh explorer
            </button>
            {graph && (
              <Link href="/playground" className="button primary">
                Trace a prompt <ArrowRight size={15} />
              </Link>
            )}
          </>
        }
      />
      <ErrorNotice message={error} />
      {loading ? (
        <Loading label="Opening model explorer" />
      ) : !graph && error ? (
        <Panel
          title="Explorer temporarily unavailable"
          subtitle="Retry after the runtime becomes available."
        >
          <div className="panel-body">
            <button className="button secondary" onClick={refresh}>
              Retry explorer
            </button>
          </div>
        </Panel>
      ) : !graph ? (
        <EmptyState
          icon={<Network size={28} />}
          title="Your next model, under the microscope."
          href="/"
          action="Choose a model"
        >
          Load a checkpoint from the library to explore its architecture,
          inspect weight matrices and discover their geometry.
        </EmptyState>
      ) : (
        <>
          <div className="metrics-grid">
            <Metric
              label="Parameters"
              value={number(graph.total_parameters, 0)}
              note="Measured from loaded weights"
            />
            <Metric
              label="Transformer blocks"
              value={graph.metadata.num_layers}
              note={`${graph.metadata.num_heads} attention heads per block`}
            />
            <Metric
              label="Hidden dimension"
              value={graph.metadata.hidden_size}
              note={`${graph.metadata.num_kv_heads} key/value heads`}
            />
            <Metric
              label="Weight matrices"
              value={graph.matrices.length}
              note="Two dimensional parameters"
            />
          </div>
          <div className="tabs" role="tablist" aria-label="Explorer view">
            <button
              role="tab"
              aria-selected={view === "graph"}
              className={view === "graph" ? "active" : ""}
              onClick={() => setView("graph")}
            >
              Architecture graph
            </button>
            <button
              role="tab"
              aria-selected={view === "table"}
              className={view === "table" ? "active" : ""}
              onClick={() => setView("table")}
            >
              Weight inventory
            </button>
          </div>
          {view === "graph" ? (
            <div className="graph-layout">
              <Panel
                title="Model architecture"
                subtitle="Scroll to zoom, drag to pan, select a block to inspect its matrices."
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
                    ? "All weight matrices"
                    : `Block ${layer} matrices`
                }
                subtitle={`${weights.length} matrices in view`}
                action={
                  layer !== null && (
                    <button
                      className="text-link icon-button"
                      onClick={() => setLayer(null)}
                    >
                      Show all
                    </button>
                  )
                }
              >
                <div className="filter-input">
                  <input
                    aria-label="Filter matrices"
                    placeholder="Filter by matrix name…"
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
                          {w.shape.join(" × ")} · {number(w.elements, 0)}{" "}
                          elements
                        </small>
                      </div>
                      <ChevronRight size={15} />
                    </button>
                  ))}
                  {!weights.length && (
                    <div className="empty-inline">No matching matrices.</div>
                  )}
                </div>
              </Panel>
            </div>
          ) : (
            <Panel
              title="Weight inventory"
              subtitle="Select any matrix to compute its properties."
            >
              <div className="filter-input">
                <input
                  aria-label="Filter matrices"
                  placeholder="Filter by matrix name…"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                />
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>MATRIX</th>
                      <th>SHAPE</th>
                      <th>ELEMENTS</th>
                      <th>DTYPE</th>
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
                        <td>{number(w.elements, 0)}</td>
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
              <Loading label={`Computing properties of ${selected}`} />
            )}
            {metrics && (
              <MatrixResults metrics={metrics} model={graph.metadata} />
            )}
          </div>
        </>
      )}
    </>
  );
}
