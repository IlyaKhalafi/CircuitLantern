"use client";
import { useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  useReactFlow,
  useNodesInitialized,
  useStore,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { useI18n } from "@/i18n/provider";
import type { BlockFlow } from "@/lib/types";

type FlowNode = Node<
  {
    label: string;
    detail: string;
    weight: string | null;
    onInspect: (name: string) => void;
  },
  "operation"
>;
function Operation({ data }: NodeProps<FlowNode>) {
  const { t } = useI18n();
  return (
    <div className={"flow-operation " + (data.weight ? "flow-weight" : "")}>
      <Handle type="target" position={Position.Top} id="top" />
      <Handle type="target" position={Position.Left} id="side-in" />
      {data.weight ? (
        <button
          className="nodrag"
          title={data.weight}
          aria-label={t("Inspect {name}", { name: data.weight })}
          onClick={(event) => {
            event.stopPropagation();
            data.onInspect(data.weight!);
          }}
        >
          <strong>{t(data.label)}</strong>
          <small>{data.detail}</small>
        </button>
      ) : (
        <>
          <strong>{t(data.label)}</strong>
          {data.detail && <small>{t(data.detail)}</small>}
        </>
      )}
      <Handle type="source" position={Position.Bottom} id="bottom" />
      <Handle type="source" position={Position.Left} id="side-out" />
    </div>
  );
}
const nodeTypes = { operation: Operation };
function FitComputation({
  flow,
  section,
}: {
  flow: BlockFlow;
  section: string;
}) {
  const initialized = useNodesInitialized();
  const { fitView } = useReactFlow();
  const width = useStore((state) => state.width),
    height = useStore((state) => state.height);
  useEffect(() => {
    if (!initialized || !width || !height) return;
    const mlp = new Set([
      "norm-mlp",
      "up",
      "gate",
      "act",
      "multiply",
      "down",
      "output",
    ]);
    const selected =
      section === "all"
        ? flow.nodes
        : flow.nodes.filter((node) =>
            section === "attention"
              ? !mlp.has(node.id)
              : mlp.has(node.id) ||
                node.id === (flow.parallel_residual ? "input" : "add-attn"),
          );
    fitView({
      nodes: selected.map((node) => ({ id: node.id })),
      padding: 0.14,
      maxZoom: 1.2,
      duration: 0,
    });
  }, [flow, section, initialized, width, height, fitView]);
  return null;
}
export function BlockDataflow({
  flow,
  onInspect,
}: {
  flow?: BlockFlow;
  onInspect: (name: string) => void;
}) {
  const { t } = useI18n();
  const [section, setSection] = useState("attention");
  const { nodes, edges } = useMemo(
    () => ({
      nodes: (flow?.nodes ?? []).map((node) => ({
        id: node.id,
        type: "operation" as const,
        position: { x: node.x, y: node.y },
        data: {
          label: node.label,
          detail: node.detail,
          weight: node.weight,
          onInspect,
        },
      })),
      edges: (flow?.edges ?? []).map((edge, index) => ({
        id: String(index),
        source: edge.source,
        target: edge.target,
        label: t(edge.label),
        sourceHandle: edge.residual ? "side-out" : "bottom",
        targetHandle: edge.residual ? "side-in" : "top",
        type: "smoothstep",
        style: {
          stroke: edge.residual ? "var(--accent)" : "var(--blue)",
          strokeWidth: edge.residual ? 2 : 1.5,
        },
        labelStyle: { fill: "var(--ink)", fontSize: 10 },
        labelBgStyle: { fill: "var(--panel)" },
        labelBgPadding: [5, 3] as [number, number],
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edge.residual ? "var(--accent)" : "var(--blue)",
        },
      })),
    }),
    [flow, onInspect, t],
  );
  if (!flow?.supported)
    return (
      <div className="empty-inline">
        {t(
          "Detailed block dataflow is not available for this architecture yet. Use the weight inventory to inspect its matrices.",
        )}
      </div>
    );
  return (
    <>
      <p className="flow-intro">
        {t(
          "Follow the arrows to see where each output goes. Select a matrix to inspect it.",
        )}
        <span>
          {flow.parallel_residual
            ? t("Parallel attention and MLP branches")
            : t("Sequential attention and MLP branches")}
        </span>
      </p>
      <div className="flow-focus">
        <label>
          {t("Diagram focus")}
          <select
            aria-label={t("Diagram focus")}
            value={section}
            onChange={(event) => setSection(event.target.value)}
          >
            <option value="all">{t("Full block")}</option>
            <option value="attention">{t("Attention path")}</option>
            <option value="mlp">{t("MLP path")}</option>
          </select>
        </label>
        <span>
          {t(
            "Blue arrows carry activations. Gold arrows carry the residual stream.",
          )}
        </span>
      </div>
      <div className="graph-container block-dataflow" dir="ltr">
        <ReactFlow<FlowNode>
          key={flow.layer}
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          minZoom={0.2}
          maxZoom={2}
          onNodeClick={(_, node) => {
            if (node.data.weight) onInspect(node.data.weight);
          }}
          nodesDraggable={false}
          nodesConnectable={false}
          nodesFocusable={false}
          edgesFocusable={false}
          elementsSelectable={false}
          deleteKeyCode={null}
          aria-label={t("Transformer block computation flow")}
        >
          <FitComputation flow={flow} section={section} />
          <Background color="var(--line)" gap={20} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <details className="flow-description">
        <summary>{t("Read the computation and scope")}</summary>
        <ol>
          {flow.nodes.map((node) => (
            <li key={node.id}>
              <strong>{t(node.label)}</strong>
              {node.weight ? (
                <button
                  className="text-link"
                  onClick={() => onInspect(node.weight!)}
                >
                  {node.weight}
                </button>
              ) : (
                <span>{t(node.detail)}</span>
              )}
              <small>
                {t("Output goes to {targets}", {
                  targets:
                    flow.edges
                      .filter((edge) => edge.source === node.id)
                      .map((edge) =>
                        t(
                          flow.nodes.find((item) => item.id === edge.target)
                            ?.label || edge.target,
                        ),
                      )
                      .join(" · ") || t("Next block"),
                })}
              </small>
            </li>
          ))}
        </ol>
        <p>
          {t(
            "This is an inference computation diagram. Normalization, positional encoding and head operations are summarized. It shows connectivity, not measured causal attribution.",
          )}
        </p>
      </details>
    </>
  );
}
