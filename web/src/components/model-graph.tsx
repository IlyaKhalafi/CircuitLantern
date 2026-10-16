"use client";
import { useI18n } from "@/i18n/provider";
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  type NodeProps,
  type Node,
  useNodesInitialized,
  useNodesState,
  useReactFlow,
  useStore,
} from "@xyflow/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { BlockDataflow } from "@/components/block-dataflow";
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
  const { t } = useI18n();
  return (
    <div className={`graph-node ${data.selected ? "selected" : ""}`}>
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: "var(--blue)", border: 0 }}
      />
      <div className="graph-node-header">
        {t(data.label)}
        <span>{t(data.subtitle)}</span>
      </div>
      <div className="graph-matrices">
        {data.matrices.map((weight) => (
          <button
            key={weight.name}
            type="button"
            className={`graph-matrix nodrag nowheel ${weight.name.includes(".mlp.") ? "mlp" : ""}`}
            title={`${weight.name} · ${weight.shape.join(" × ")}${data.sharedOutput ? " " + t("· shared with token embedding") : ""}`}
            aria-label={t("Inspect {v0}, shape {v1}{v2}", {
              v0: weight.name,
              v1: weight.shape.join(" by "),
              v2: data.sharedOutput ? t(", shared output projection") : "",
            })}
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
          {t("Shared or specialized output projection")}
        </span>
      )}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "var(--blue)", border: 0 }}
      />
    </div>
  );
}
const nodeTypes = { layer: Layer };
export function getLayer(name: string): number | null {
  const match = name.match(/(?:layers|h)\.(\d+)\./);
  return match ? +match[1] : null;
}
function FitGraphViewport({
  selectedLayer,
  modelKey,
  layoutReady,
}: {
  selectedLayer: number | null;
  modelKey: string;
  layoutReady: boolean;
}) {
  const { fitView } = useReactFlow();
  const initialized = useNodesInitialized();
  const width = useStore((state) => state.width);
  const height = useStore((state) => state.height);
  const fitted = useRef(false);
  const lastFit = useRef("");
  useEffect(() => {
    if (!initialized || !layoutReady || !width || !height) return;
    const frame = JSON.stringify([selectedLayer, modelKey, width, height]);
    if (lastFit.current === frame) return;
    lastFit.current = frame;
    const duration =
      !fitted.current ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? 0
        : 250;
    fitted.current = true;
    fitView({
      ...(selectedLayer !== null
        ? { nodes: [{ id: String(selectedLayer) }] }
        : {}),
      padding: selectedLayer === null ? 0.12 : 0.65,
      maxZoom: selectedLayer === null ? 0.9 : 1.4,
      duration,
    });
  }, [
    fitView,
    initialized,
    layoutReady,
    width,
    height,
    selectedLayer,
    modelKey,
  ]);
  return null;
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
  const { t, rich } = useI18n();
  const [mode, setMode] = useState("overview");
  useEffect(() => {
    if (selectedLayer === null && mode === "flow") setMode("overview");
  }, [selectedLayer, mode]);
  const { nodes: plannedNodes, edges } = useMemo(() => {
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
          subtitle: t("{count} TOKENS", {
            count: graph.metadata.vocab_size.toLocaleString("en"),
          }),
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
          subtitle: t("{count} MATRICES", { count: matrices.length }),
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
        style: { stroke: "var(--blue-border)", strokeWidth: 1.5 },
      })),
    };
  }, [graph, selectedLayer, onInspect, t]);
  const [nodes, setNodes, onNodesChange] = useNodesState(plannedNodes);
  useEffect(() => {
    setNodes((current) => {
      if (current === plannedNodes) return current;
      const previous = new Map(current.map((node) => [node.id, node]));
      return plannedNodes.map((node) => ({
        ...node,
        measured: previous.get(node.id)?.measured,
        position: previous.get(node.id)?.position ?? node.position,
      }));
    });
  }, [plannedNodes, setNodes]);
  const stackedNodes = useMemo(() => {
    if (nodes.some((node) => !node.measured?.height)) return null;
    let y = 0;
    return nodes.map((node) => {
      const position = { x: 0, y };
      y += node.measured!.height! + 28;
      return node.position.x === 0 && node.position.y === position.y
        ? node
        : { ...node, position };
    });
  }, [nodes]);
  const layoutReady =
    stackedNodes !== null &&
    stackedNodes.every((node, index) => node === nodes[index]);
  useEffect(() => {
    if (stackedNodes && !layoutReady) setNodes(stackedNodes);
  }, [stackedNodes, layoutReady, setNodes]);
  return (
    <>
      <div className="graph-toolbar">
        <label>
          {t("Focus on")}
          <select
            aria-label={t("Focus on transformer block")}
            value={selectedLayer === null ? "all" : selectedLayer}
            onChange={(event) =>
              onSelect(
                event.target.value === "all" ? null : +event.target.value,
              )
            }
          >
            {mode === "overview" && (
              <option value="all">{t("Full model overview")}</option>
            )}
            {Array.from({ length: graph.metadata.num_layers }, (_, i) => (
              <option key={i} value={i}>
                {rich("Transformer block {v0}", {
                  v0: i,
                })}
              </option>
            ))}
          </select>
        </label>
        <div
          className="flow-view-switch"
          role="group"
          aria-label={t("Architecture detail")}
        >
          <button
            className={mode === "overview" ? "active" : ""}
            aria-pressed={mode === "overview"}
            onClick={() => setMode("overview")}
          >
            {t("Overview")}
          </button>
          <button
            className={mode === "flow" ? "active" : ""}
            aria-pressed={mode === "flow"}
            onClick={() => {
              if (selectedLayer === null) onSelect(0);
              setMode("flow");
            }}
          >
            {t("Block dataflow")}
          </button>
        </div>
      </div>
      {mode === "flow" ? (
        <BlockDataflow
          flow={graph.dataflow?.find(
            (item) => item.layer === (selectedLayer ?? 0),
          )}
          onInspect={onInspect}
        />
      ) : (
        <div className="graph-container">
          <ReactFlow<LayerNode>
            nodes={nodes}
            onNodesChange={onNodesChange}
            deleteKeyCode={null}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodeClick={(_, node) =>
              onSelect(/^\d+$/.test(node.id) ? +node.id : null)
            }
            minZoom={0.05}
            maxZoom={2}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            aria-label={t("Interactive model architecture")}
          >
            <FitGraphViewport
              layoutReady={layoutReady}
              selectedLayer={selectedLayer}
              modelKey={
                graph.metadata.execution_id ??
                `${graph.metadata.id}@${graph.metadata.revision}`
              }
            />
            <Background color="var(--line)" gap={20} size={1} />
            <Controls showInteractive={false} />
            <MiniMap
              nodeColor={(node) =>
                node.data.selected ? "var(--accent)" : "var(--blue)"
              }
              maskColor="var(--graph-mask)"
              style={{ width: 90, height: 110 }}
            />
          </ReactFlow>
        </div>
      )}
    </>
  );
}
