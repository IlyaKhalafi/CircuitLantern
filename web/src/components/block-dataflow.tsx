"use client";
import { useEffect, useMemo, useState } from "react";
import {
  Background,
  BaseEdge,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  useReactFlow,
  useStore,
  type Node,
  type NodeProps,
  type EdgeProps,
} from "@xyflow/react";
import { useI18n } from "@/i18n/provider";
import type { BlockFlow } from "@/lib/types";

type FlowNode = Node<
  {
    label: string;
    detail: string;
    weight: string | null;
    onInspect: (name: string) => void;
    opaque?: boolean;
  },
  "operation"
>;
function Operation({ data }: NodeProps<FlowNode>) {
  const { t } = useI18n();
  return (
    <div
      className={
        "flow-operation " +
        (data.weight ? "flow-weight" : "") +
        (data.opaque ? " flow-opaque" : "")
      }
    >
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
      <Handle type="source" position={Position.Right} id="branch-out" />
    </div>
  );
}
const nodeTypes = { operation: Operation };
function ResidualEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  markerEnd,
  style,
  label,
  data,
}: EdgeProps) {
  const lane = typeof data?.lane === "number" ? data.lane : -55,
    bend = Math.min(12, Math.abs(targetY - sourceY) / 3);
  const path = `M ${sourceX},${sourceY} L ${lane + bend},${sourceY} Q ${lane},${sourceY} ${lane},${sourceY + bend} L ${lane},${targetY - bend} Q ${lane},${targetY} ${lane + bend},${targetY} L ${targetX},${targetY}`;
  return (
    <BaseEdge
      id={id}
      path={path}
      markerEnd={markerEnd}
      style={style}
      label={label}
      labelX={lane}
      labelY={(sourceY + targetY) / 2}
      labelStyle={{ fill: "var(--ink)", fontSize: 10 }}
      labelBgStyle={{ fill: "var(--panel)" }}
      labelBgPadding={[5, 3]}
    />
  );
}
function SkipEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  markerEnd,
  style,
  data,
}: EdgeProps) {
  const lane = typeof data?.lane === "number" ? data.lane : sourceX + 100;
  const path = `M ${sourceX},${sourceY} L ${sourceX},${sourceY + 12} L ${lane},${sourceY + 12} L ${lane},${targetY - 16} L ${targetX},${targetY - 16} L ${targetX},${targetY}`;
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
}
const edgeTypes = { residual: ResidualEdge, skip: SkipEdge };
function FitComputation({
  flow,
  section,
}: {
  flow: BlockFlow;
  section: string;
}) {
  const { fitBounds, getNode, setViewport, viewportInitialized } =
    useReactFlow();
  const measurements = useStore((state) =>
    flow.nodes
      .map((node) => state.nodeLookup.get(node.id)?.measured?.height ?? 0)
      .join(","),
  );
  const width = useStore((state) => state.width),
    height = useStore((state) => state.height);
  useEffect(() => {
    if (!viewportInitialized || !width || !height) return;
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
              ? node.section === "attention" ||
                (!flow.provenance && !mlp.has(node.id))
              : node.section === "mlp" ||
                mlp.has(node.id) ||
                node.id === (flow.parallel_residual ? "input" : "add-attn"),
          );
    if (!selected.length) return;
    const left = Math.min(-90, ...flow.nodes.map((node) => node.x - 70));
    const right = Math.max(...selected.map((node) => node.x + 260));
    const top = Math.min(...selected.map((node) => node.y));
    const bottom = Math.max(
      ...selected.map(
        (node) => node.y + (getNode(node.id)?.measured?.height || 80),
      ),
    );
    if (width < 600) {
      const zoom = 0.75;
      setViewport(
        {
          x: width / 2 - ((left + right) / 2) * zoom,
          y: 28 - top * zoom,
          zoom,
        },
        { duration: 0 },
      );
      return;
    }
    fitBounds(
      { x: left, y: top, width: right - left, height: bottom - top },
      { padding: 0.13, duration: 0 },
    );
  }, [
    flow,
    section,
    viewportInitialized,
    measurements,
    width,
    height,
    fitBounds,
    getNode,
    setViewport,
  ]);
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
  const [section, setSection] = useState(
    flow?.provenance && !flow.nodes.some((node) => node.section === "attention")
      ? "all"
      : "attention",
  );
  const [focused, setFocused] = useState("");
  useEffect(() => {
    setFocused("");
    setSection(
      flow?.provenance &&
        !flow.nodes.some((node) => node.section === "attention")
        ? "all"
        : "attention",
    );
  }, [flow]);
  const connected = useMemo(() => {
    if (!focused || !flow) return null;
    const result = new Set([focused]);
    for (const direction of ["up", "down"]) {
      const visited = new Set([focused]);
      const queue = [focused];
      while (queue.length) {
        const current = queue.pop()!;
        for (const edge of flow.edges) {
          const next =
            direction === "up"
              ? edge.target === current
                ? edge.source
                : null
              : edge.source === current
                ? edge.target
                : null;
          if (next && !visited.has(next)) {
            visited.add(next);
            result.add(next);
            queue.push(next);
          }
        }
      }
    }
    return result;
  }, [flow, focused]);
  const { nodes, edges } = useMemo(
    () => ({
      nodes: (flow?.nodes ?? []).map((node) => ({
        id: node.id,
        type: "operation" as const,
        position: { x: node.x, y: node.y },
        className: connected && !connected.has(node.id) ? "flow-dimmed" : "",

        data: {
          label: node.label,
          detail: node.detail,
          weight: node.weight,
          opaque: node.opaque,
          onInspect,
        },
      })),
      edges: (flow?.edges ?? []).map((edge, index) => ({
        id: String(index),
        source: edge.source,
        target: edge.target,
        label: t(edge.label),
        sourceHandle: edge.branch
          ? "branch-out"
          : edge.residual
            ? "side-out"
            : "bottom",
        targetHandle: edge.residual ? "side-in" : "top",
        type: edge.residual
          ? "residual"
          : flow?.provenance &&
              (flow.nodes.find((node) => node.id === edge.target)?.y ?? 0) -
                (flow.nodes.find((node) => node.id === edge.source)?.y ?? 0) >
                115
            ? "skip"
            : "smoothstep",
        data: {
          lane: edge.residual
            ? Math.min(-55, ...(flow?.nodes ?? []).map((node) => node.x - 55))
            : Math.max(...(flow?.nodes ?? []).map((node) => node.x + 205)) +
              (index % 4) * 12,
        },
        style: {
          stroke: edge.residual ? "var(--accent)" : "var(--blue)",
          strokeWidth: edge.residual ? 2 : 1.5,
          opacity:
            connected &&
            (!connected.has(edge.source) || !connected.has(edge.target))
              ? 0.12
              : 1,
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
    [flow, onInspect, t, connected],
  );
  if (!flow?.supported)
    return (
      <div className="empty-inline">
        {t(
          "Connections could not be resolved safely for this block. Inspect its weights in the inventory.",
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
          {flow.provenance
            ? t("Connections derived from model source")
            : flow.parallel_residual
              ? t("Parallel attention and MLP branches")
              : t("Sequential attention and MLP branches")}
        </span>
      </p>
      {flow.provenance && (
        <div className="flow-coverage" role="status">
          <strong>
            {t("{mapped} of {total} matrices connected", {
              mapped: flow.mapped_matrices ?? 0,
              total: flow.total_matrices ?? 0,
            })}
          </strong>
          <span>{t("Uncached inference")}</span>
          {!!flow.opaque_modules?.length && (
            <span>{t("Some module internals remain summarized")}</span>
          )}
        </div>
      )}
      <div className="flow-focus">
        <label>
          {t("Diagram focus")}
          <select
            aria-label={t("Diagram focus")}
            value={section}
            onChange={(event) => setSection(event.target.value)}
          >
            <option value="all">{t("Full block")}</option>
            {(!flow.provenance ||
              flow.nodes.some((node) => node.section === "attention")) && (
              <option value="attention">
                {t(
                  flow.mixer_type === "linear_attention"
                    ? "Token mixer path"
                    : "Attention path",
                )}
              </option>
            )}
            {(!flow.provenance ||
              flow.nodes.some((node) => node.section === "mlp")) && (
              <option value="mlp">{t("MLP path")}</option>
            )}
          </select>
        </label>
        <label>
          {t("Trace connections")}
          <select
            aria-label={t("Trace connections")}
            value={focused}
            onChange={(event) => setFocused(event.target.value)}
          >
            <option value="">{t("All connections")}</option>
            {flow.nodes.map((node) => (
              <option key={node.id} value={node.id}>
                {t(node.label)}
                {node.weight
                  ? " · " + node.weight.split(".").slice(-2, -1)[0]
                  : ""}
              </option>
            ))}
          </select>
        </label>
        <span>
          {t(
            "Blue arrows carry activations. Gold arrows carry the residual stream.",
          )}
        </span>
      </div>
      <p className="flow-mobile-hint">
        {t("Pan to follow branches or zoom out for an overview.")}
      </p>
      <div className="graph-container block-dataflow" dir="ltr">
        <ReactFlow<FlowNode>
          key={flow.layer}
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
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
      {!!flow.unmapped_matrices?.length && (
        <details className="flow-description">
          <summary>{t("Unmapped matrices")}</summary>
          <p>
            {t(
              "These weights were not mapped to the selected output path. They remain available for inspection.",
            )}
          </p>
          <div className="flow-unmapped">
            {flow.unmapped_matrices.map((name) => (
              <button
                className="text-link"
                key={name}
                onClick={() => onInspect(name)}
              >
                {name}
              </button>
            ))}
          </div>
        </details>
      )}
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
        {flow.provenance && (
          <p>
            {t(
              "Module connections come from installed forward source. Verified mixer diagrams summarize kernel operations. Hooks and decorators can change execution.",
            )}
          </p>
        )}
        <p>
          {t(
            "This is an inference computation diagram. Normalization, positional encoding and head operations are summarized. It shows connectivity, not measured causal attribution.",
          )}
        </p>
      </details>
    </>
  );
}
