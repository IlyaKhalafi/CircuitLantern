"use client";

import {
  ArrowRight,
  ChevronRight,
  Download,
  Layers3,
  Network,
  Search,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Heatmap, Histogram, SeriesChart } from "@/components/charts";
import { getLayer, ModelGraphView } from "@/components/model-graph";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  Metric,
  PageHeader,
  Panel,
} from "@/components/ui";
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
  const detail = useRef<HTMLDivElement>(null);
  useEffect(() => {
    api<RuntimeState>("/runtime")
      .then((state) =>
        state.loaded ? api<ModelGraph>("/graph").then(setGraph) : null,
      )
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);
  async function inspect(name: string) {
    const id = ++requestId.current;
    setSelected(name);
    setMetrics(null);
    setAnalyzing(true);
    setError(null);
    try {
      const result = await api<MatrixMetrics>(
        `/matrix?name=${encodeURIComponent(name)}`,
      );
      if (id === requestId.current) {
        setMetrics(result);
        setTimeout(
          () =>
            detail.current?.scrollIntoView({
              behavior: "smooth",
              block: "start",
            }),
          50,
        );
      }
    } catch (e) {
      if (id === requestId.current) setError((e as Error).message);
    } finally {
      if (id === requestId.current) setAnalyzing(false);
    }
  }
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
          graph && (
            <Link href="/playground" className="button primary">
              Trace a prompt <ArrowRight size={15} />
            </Link>
          )
        }
      />
      <ErrorNotice message={error} />
      {loading ? (
        <Loading label="Opening model explorer" />
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
          <div ref={detail} className="section-gap">
            {analyzing && (
              <Loading label={`Computing properties of ${selected}`} />
            )}
            {metrics && (
              <>
                <div className="detail-header">
                  <div>
                    <div className="eyebrow">MATRIX INSPECTOR</div>
                    <h2>{metrics.name}</h2>
                    <p>
                      {metrics.shape.join(" × ")} · {metrics.dtype} ·{" "}
                      {number(metrics.elements, 0)} elements
                    </p>
                  </div>
                  <button
                    className="button secondary"
                    onClick={() =>
                      downloadJSON(
                        {
                          model_id: graph.metadata.id,
                          revision: graph.metadata.revision,
                          ...metrics,
                        },
                        "matrix-analysis.json",
                      )
                    }
                  >
                    <Download size={14} /> Export analysis
                  </button>
                </div>
                {!metrics.spectrum_exact && (
                  <div className="notice info">
                    Spectral norm and stable rank use a randomized SVD on the
                    full matrix. The leading 64 singular values are approximate.
                    Cumulative energy is measured against the full matrix
                    Frobenius norm. The condition number is unavailable without
                    an exact spectrum.
                  </div>
                )}
                <div className="two-column">
                  <Panel
                    title="Weight landscape"
                    subtitle="Region means reveal structure without listing individual elements."
                  >
                    <Heatmap
                      data={metrics.heatmap}
                      label={metrics.heatmap_method}
                    />
                  </Panel>
                  <Panel
                    title="Singular value spectrum"
                    subtitle={
                      metrics.spectrum_exact
                        ? "Exact spectrum · first 128 singular values"
                        : "Full matrix randomized SVD · leading 64 components"
                    }
                  >
                    <SeriesChart
                      data={metrics.singular_values.map((v, i) => ({
                        index: i + 1,
                        value: v,
                      }))}
                      series={[
                        {
                          key: "value",
                          label: "Singular value",
                          color: "#bd906b",
                        },
                      ]}
                      area
                    />
                  </Panel>
                </div>
                <Panel
                  title="Matrix properties"
                  subtitle="Spectral radius is defined for square matrices and computed exactly up to 768 × 768."
                >
                  <div className="weight-stats">
                    {[
                      {
                        label: "Frobenius norm",
                        value: metrics.frobenius_norm,
                      },
                      { label: "Spectral norm", value: metrics.spectral_norm },
                      {
                        label: "Spectral radius",
                        value: metrics.spectral_radius,
                      },
                      { label: "Stable rank", value: metrics.stable_rank },
                      {
                        label: "Condition number",
                        value: metrics.condition_number,
                      },
                      { label: "L1 norm", value: metrics.l1_norm },
                      { label: "Infinity norm", value: metrics.infinity_norm },
                      {
                        label: "Exact zeros",
                        value: `${number(metrics.sparsity * 100)}%`,
                      },
                      { label: "Mean", value: metrics.mean },
                      { label: "Standard deviation", value: metrics.std },
                      { label: "Minimum", value: metrics.min },
                      { label: "Maximum", value: metrics.max },
                    ].map((m) => (
                      <div className="weight-stat" key={m.label}>
                        <span>{m.label}</span>
                        <strong>
                          {typeof m.value === "string"
                            ? m.value
                            : number(m.value, 5)}
                        </strong>
                      </div>
                    ))}
                  </div>
                </Panel>
                <div className="two-column section-gap">
                  <Panel
                    title="Weight distribution"
                    subtitle={`${metrics.histogram_sampled ? "Subsampled" : "All"} elements · 48 histogram bins`}
                  >
                    <Histogram data={metrics.histogram} />
                  </Panel>
                  <Panel
                    title="Cumulative spectral energy"
                    subtitle="Fraction of squared singular values captured by the leading components."
                  >
                    <SeriesChart
                      data={metrics.cumulative_energy.map((v, i) => ({
                        index: i + 1,
                        value: v * 100,
                      }))}
                      series={[
                        {
                          key: "value",
                          label: "Energy percent",
                          color: "#87a38e",
                        },
                      ]}
                      area
                    />
                  </Panel>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </>
  );
}
