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
} from "@xyflow/react";
import { useMemo } from "react";
import type { ModelGraph } from "@/lib/types";
import "@xyflow/react/dist/style.css";

type LayerNode = Node<
  { label: string; subtitle: string; matrices: string[]; selected: boolean },
  "layer"
>;
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
        {data.matrices.map((name, i) => (
          <span
            className={`graph-matrix ${name.includes("mlp") || name.includes("fc") ? "mlp" : ""}`}
            key={i}
          >
            {name.replace("_proj", "")}
          </span>
        ))}
      </div>
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
}: {
  graph: ModelGraph;
  selectedLayer: number | null;
  onSelect: (layer: number | null) => void;
}) {
  const { nodes, edges } = useMemo(() => {
    const nodes: LayerNode[] = [
      {
        id: "embedding",
        type: "layer",
        position: { x: 0, y: 0 },
        data: {
          label: "Token embedding",
          subtitle: `${graph.metadata.vocab_size.toLocaleString()} × ${graph.metadata.hidden_size}`,
          matrices: ["W_embed"],
          selected: false,
        },
      },
    ];
    for (let i = 0; i < graph.metadata.num_layers; i++) {
      const matrices = graph.matrices
        .filter((w) => getLayer(w.name) === i)
        .map((w) => w.name.split(".").at(-2) || "W");
      nodes.push({
        id: String(i),
        type: "layer",
        position: { x: 0, y: (i + 1) * 130 },
        data: {
          label: `Transformer block ${String(i).padStart(2, "0")}`,
          subtitle: `${matrices.length} MATRICES`,
          matrices,
          selected: selectedLayer === i,
        },
      });
    }
    nodes.push({
      id: "unembedding",
      type: "layer",
      position: { x: 0, y: (graph.metadata.num_layers + 1) * 130 },
      data: {
        label: "Final norm → unembedding",
        subtitle: "LOGITS",
        matrices: ["W_U"],
        selected: false,
      },
    });
    const edges = nodes.slice(1).map((n, i) => ({
      id: `e${i}`,
      source: nodes[i].id,
      target: n.id,
      style: { stroke: "#becbda", strokeWidth: 1.5 },
      animated: false,
    }));
    return { nodes, edges };
  }, [graph, selectedLayer]);
  return (
    <div className="graph-container">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) =>
          onSelect(/^\d+$/.test(node.id) ? +node.id : null)
        }
        fitView
        fitViewOptions={{ maxZoom: 0.9, minZoom: 0.18 }}
        minZoom={0.08}
        maxZoom={2}
        nodesDraggable={false}
        nodesConnectable={false}
        proOptions={{ hideAttribution: false }}
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
  );
}
