"use client";

import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  type NodeProps,
  type Node,
  type ReactFlowInstance,
} from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import type { ModelGraph, Weight } from "@/lib/types";
import "@xyflow/react/dist/style.css";

type LayerNode = Node<
  {
    label: string;
    subtitle: string;
    matrices: Weight[];
    selected: boolean;
    sharedOutput?: boolean;
    onInspect: (name: string) => void;
  },
  "layer"
>;

const projectionLabels: Record<string, string> = {
  q_proj: "W_Q",
  k_proj: "W_K",
  v_proj: "W_V",
  o_proj: "W_O",
  c_attn: "QKV",
  query_key_value: "QKV",
  c_fc: "MLP in",
  dense_h_to_4h: "MLP in",
  dense_4h_to_h: "MLP out",
  wte: "W_E",
  embed_tokens: "W_E",
  embed_in: "W_E",
  lm_head: "W_U",
  embed_out: "W_U",
};

function matrixLabel(name: string) {
  const component = name.split(".").at(-2) || "W";
  if (component === "c_proj")
    return name.includes(".attn.") ? "W_O" : "MLP out";
  return projectionLabels[component] || component.replace("_proj", "");
}

function Layer({ data }: NodeProps<LayerNode>) {
  return (
    <div className={`graph-node ${data.selected ? "selected" : ""}`}>
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: "#b8c3d1", border: 0 }}
      />
      <div className="graph-node-header">
        {data.label}
        <span>{data.subtitle}</span>
      </div>
      <div className="graph-matrices">
        {data.matrices.map((weight) => (
          <button
            key={weight.name}
            type="button"
            className={`graph-matrix nodrag nowheel ${weight.name.includes(".mlp.") ? "mlp" : ""}`}
            title={`${weight.name} · ${weight.shape.join(" × ")}${data.sharedOutput ? " · shared with token embedding" : ""}`}
            aria-label={`Inspect ${weight.name}, shape ${weight.shape.join(" by ")}${data.sharedOutput ? ", shared output projection" : ""}`}
            onClick={(event) => {
              event.stopPropagation();
              data.onInspect(weight.name);
            }}
          >
            <strong>
              {data.sharedOutput ? "W_U" : matrixLabel(weight.name)}
            </strong>
            <small>{weight.shape.join("×")}</small>
          </button>
        ))}
      </div>
      {!data.matrices.length && (
        <span className="graph-output-note">
          Shared or specialized output projection
        </span>
      )}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "#b8c3d1", border: 0 }}
      />
    </div>
  );
}
const nodeTypes = { layer: Layer };

export function getLayer(name: string): number | null {
  const match = name.match(/(?:layers|h)\.(\d+)\./);
  return match ? +match[1] : null;
}

export function ModelGraphView({
  graph,
  selectedLayer,
  onSelect,
  onInspect,
}: {
  graph: ModelGraph;
  selectedLayer: number | null;
  onSelect: (layer: number | null) => void;
  onInspect: (name: string) => void;
}) {
  const [instance, setInstance] = useState<ReactFlowInstance<LayerNode> | null>(
    null,
  );
  const { nodes, edges } = useMemo(() => {
    const embedding = graph.matrices.filter((w) =>
      /(?:wte|embed_tokens|embed_in)\.weight$/.test(w.name),
    );
    const output = graph.matrices.filter((w) =>
      /(?:lm_head|embed_out)\.weight$/.test(w.name),
    );
    const sharedOutput = graph.metadata.tied_embeddings && !output.length;
    const nodes: LayerNode[] = [
      {
        id: "embedding",
        type: "layer",
        position: { x: 0, y: 0 },
        data: {
          label: "Token embedding",
          subtitle: `${graph.metadata.vocab_size.toLocaleString()} TOKENS`,
          matrices: embedding,
          selected: false,
          onInspect,
        },
      },
    ];
    for (let i = 0; i < graph.metadata.num_layers; i++) {
      const matrices = graph.matrices.filter((w) => getLayer(w.name) === i);
      nodes.push({
        id: String(i),
        type: "layer",
        position: { x: 0, y: (i + 1) * 150 },
        data: {
          label: `Transformer block ${String(i).padStart(2, "0")}`,
          subtitle: `${matrices.length} MATRICES`,
          matrices,
          selected: selectedLayer === i,
          onInspect,
        },
      });
    }
    nodes.push({
      id: "unembedding",
      type: "layer",
      position: { x: 0, y: (graph.metadata.num_layers + 1) * 150 },
      data: {
        label: "Final norm → unembedding",
        subtitle: sharedOutput ? "LOGITS · SHARED W_E" : "LOGITS",
        matrices: sharedOutput ? embedding : output,
        sharedOutput,
        selected: false,
        onInspect,
      },
    });
    return {
      nodes,
      edges: nodes.slice(1).map((node, index) => ({
        id: `e${index}`,
        source: nodes[index].id,
        target: node.id,
        style: { stroke: "#becbda", strokeWidth: 1.5 },
      })),
    };
  }, [graph, selectedLayer, onInspect]);

  useEffect(() => {
    if (!instance) return;
    const duration = window.matchMedia("(prefers-reduced-motion: reduce)")
      .matches
      ? 0
      : 250;
    instance.fitView({
      ...(selectedLayer !== null
        ? { nodes: [{ id: String(selectedLayer) }] }
        : {}),
      padding: selectedLayer === null ? 0.12 : 0.65,
      maxZoom: selectedLayer === null ? 0.9 : 1.4,
      duration,
    });
  }, [instance, selectedLayer]);

  return (
    <>
      <div className="graph-toolbar">
        <label>
          Focus on
          <select
            aria-label="Focus on transformer block"
            value={selectedLayer === null ? "all" : selectedLayer}
            onChange={(event) =>
              onSelect(
                event.target.value === "all" ? null : +event.target.value,
              )
            }
          >
            <option value="all">Full model overview</option>
            {Array.from({ length: graph.metadata.num_layers }, (_, i) => (
              <option key={i} value={i}>
                Transformer block {i}
              </option>
            ))}
          </select>
        </label>
        <span>Select a matrix to inspect its weights</span>
      </div>
      <div className="graph-container">
        <ReactFlow<LayerNode>
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onInit={setInstance}
          onNodeClick={(_, node) =>
            onSelect(/^\d+$/.test(node.id) ? +node.id : null)
          }
          fitView
          fitViewOptions={{ maxZoom: 0.9, minZoom: 0.08 }}
          minZoom={0.05}
          maxZoom={2}
          nodesDraggable={false}
          nodesConnectable={false}
          aria-label="Interactive model architecture"
        >
          <Background color="#d7dfe9" gap={20} size={1} />
          <Controls showInteractive={false} />
          <MiniMap
            nodeColor={(node) => (node.data.selected ? "#dba575" : "#d6dfe9")}
            maskColor="#f5f7f9b0"
            style={{ width: 90, height: 110 }}
          />
        </ReactFlow>
      </div>
    </>
  );
}
