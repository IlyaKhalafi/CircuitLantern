"use client";
import { useMemo, useState } from "react";
import {
  Background,
  Controls,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useI18n } from "@/i18n/provider";
import { number } from "@/lib/api";
import type { FeatureCircuit } from "@/lib/features";

export function FeatureCircuitGraph({ result }: { result: FeatureCircuit }) {
  const { t } = useI18n();
  const [edgeLimit, setEdgeLimit] = useState(2);

  const { nodes, edges } = useMemo(() => {
    const layers = [...new Set(result.nodes.map((node) => node.layer))].sort(
      (a, b) => a - b,
    );
    const count = new Map<number, number>();
    const nodes: Node[] = result.nodes.map((node) => {
      const row = count.get(node.layer) || 0;
      count.set(node.layer, row + 1);
      return {
        id: node.id,
        position: { x: layers.indexOf(node.layer) * 240, y: row * 110 },
        data: {
          label: (
            <>
              <small>{t("Layer {v0}", { v0: node.layer })}</small>
              <strong>
                {node.label || t("Feature {v0}", { v0: node.feature })}
              </strong>
              <span>{number(node.mean_activation)}</span>
            </>
          ),
        },
        className: "feature-circuit-node",
        sourcePosition: "right" as Node["sourcePosition"],
        targetPosition: "left" as Node["targetPosition"],
      };
    });
    nodes.push({
      id: "output",
      position: { x: layers.length * 240, y: 100 },
      data: {
        label: (
          <>
            <small>{t("Target prediction")}</small>
            <strong dir="auto">{result.target_token || "∅"}</strong>
          </>
        ),
      },
      className: "feature-circuit-node output",
      targetPosition: "left" as Node["targetPosition"],
    });
    const maxima = new Map<string, number>();
    const destinations = new Map<string, typeof result.edges>();
    for (const edge of result.edges) {
      const value = Math.abs(
        edge.activation_change ?? edge.log_probability_change ?? 0,
      );
      maxima.set(
        edge.target,
        Math.max(maxima.get(edge.target) ?? 1e-12, value),
      );
      destinations.set(edge.target, [
        ...(destinations.get(edge.target) ?? []),
        edge,
      ]);
    }
    const visible = [...destinations.values()].flatMap((edges) =>
      [...edges]
        .sort(
          (a, b) =>
            Math.abs(b.activation_change ?? b.log_probability_change ?? 0) -
            Math.abs(a.activation_change ?? a.log_probability_change ?? 0),
        )
        .slice(0, edgeLimit || undefined),
    );
    const edges: Edge[] = visible.map((edge, index) => {
      const value = edge.activation_change ?? edge.log_probability_change ?? 0;
      return {
        id: String(index),
        source: edge.source,
        target: edge.target,
        style: {
          stroke: value >= 0 ? "var(--blue)" : "var(--red)",
          strokeWidth:
            0.8 + (3 * Math.abs(value)) / (maxima.get(edge.target) ?? 1e-12),
          opacity: 0.75,
        },
        ariaLabel: t("Measured effect {value}", { value: number(value, 5) }),
      };
    });
    return { nodes, edges };
  }, [result, t, edgeLimit]);
  return (
    <>
      <div className="feature-form">
        <label>
          {t("Edges per destination")}
          <select
            aria-label={t("Edges per destination")}
            value={edgeLimit}
            onChange={(event) => setEdgeLimit(Number(event.target.value))}
          >
            {[1, 2, 4, 0].map((value) => (
              <option value={value} key={value}>
                {value || t("All measured edges")}
              </option>
            ))}
          </select>
        </label>
        <p className="inline-note">
          {t(
            "Positive changes mean the destination increased after removing the source feature. Edge widths are scaled within each destination. Exports retain all measurements.",
          )}
        </p>
      </div>
      <div
        className="feature-circuit-canvas"
        role="region"
        aria-label={t(
          "Feature ablation graph. Numerical measurements are listed below.",
        )}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          minZoom={0.2}
          maxZoom={2.5}
          nodesDraggable={false}
          nodesFocusable={false}
          edgesFocusable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          deleteKeyCode={null}
          proOptions={{ hideAttribution: false }}
        >
          <Background color="var(--line)" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </>
  );
}
