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
import { focusBlockFlow, hasFlowSection } from "@/lib/block-flow";

type FlowNode = Node<
  {
    label: string;
    detail: string;
    weight: string | null;
    onInspect: (name: string) => void;
    opaque?: boolean;
    boundary?: "input" | "output";
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
        (data.opaque ? " flow-opaque" : "") +
        (data.boundary ? " flow-boundary" : "")
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
          {data.detail && (
            <small>
              {data.boundary
                ? t("From {operation}", { operation: t(data.detail) })
                : t(data.detail)}
            </small>
          )}
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
function FitComputation({ flow }: { flow: BlockFlow }) {
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
    const selected = flow.nodes;
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
  const [preferredSection, setSection] = useState(
    flow?.provenance && !flow.nodes.some((node) => node.section === "attention")
      ? "all"
      : "attention",
  );
  const [selection, setSelection] = useState<{
    flow: BlockFlow;
    id: string;
  } | null>(null);
  const section =
    flow && hasFlowSection(flow, preferredSection)
      ? preferredSection
      : flow?.provenance && !hasFlowSection(flow, "attention")
        ? "all"
        : "attention";
  const visibleFlow = useMemo(
    () => (flow ? focusBlockFlow(flow, section) : undefined),
    [flow, section],
  );
  const focused =
    selection?.flow === flow &&
    visibleFlow?.nodes.some((node) => node.id === selection?.id)
      ? selection!.id
      : "";
  function operationName(node: BlockFlow["nodes"][number]) {
    const origin = node.boundary
      ? flow?.nodes.find((item) => item.id === node.boundary_origin)
      : undefined;
    return origin
      ? t("{operation} · {origin}", {
          operation: t(node.label),
          origin: t(origin.label),
        })
      : t(node.label);
  }
  const connected = useMemo(() => {
    if (!focused || !visibleFlow) return null;
    const result = new Set([focused]);
    for (const direction of ["up", "down"]) {
      const visited = new Set([focused]);
      const queue = [focused];
      while (queue.length) {
        const current = queue.pop()!;
        for (const edge of visibleFlow.edges) {
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
  }, [visibleFlow, focused]);
  const { nodes, edges } = useMemo(
    () => ({
      nodes: (visibleFlow?.nodes ?? []).map((node) => ({
        id: node.id,
        type: "operation" as const,
        position: { x: node.x, y: node.y },
        className: connected && !connected.has(node.id) ? "flow-dimmed" : "",

        data: {
          label: node.label,
          detail: node.detail,
          weight: node.weight,
          opaque: node.opaque,
          boundary: node.boundary,
          onInspect,
        },
      })),
      edges: (visibleFlow?.edges ?? []).map((edge, index) => ({
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
              (visibleFlow?.nodes.find((node) => node.id === edge.target)?.y ??
                0) -
                (visibleFlow?.nodes.find((node) => node.id === edge.source)
                  ?.y ?? 0) >
                115
            ? "skip"
            : "smoothstep",
        data: {
          lane: edge.residual
            ? Math.min(
                -55,
                ...(visibleFlow?.nodes ?? []).map((node) => node.x - 55),
              )
            : Math.max(
                ...(visibleFlow?.nodes ?? []).map((node) => node.x + 205),
              ) +
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
    [flow, visibleFlow, onInspect, t, connected],
  );
  if (!flow?.supported || !visibleFlow)
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
            {t("{mapped} of {total} block matrices connected", {
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
            onChange={(event) => {
              setSection(event.target.value);
              setSelection(null);
            }}
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
            onChange={(event) =>
              setSelection(
                event.target.value ? { flow, id: event.target.value } : null,
              )
            }
          >
            <option value="">{t("All connections")}</option>
            {visibleFlow.nodes.map((node) => (
              <option key={node.id} value={node.id}>
                {operationName(node)}
                {node.weight
                  ? " · " + node.weight.split(".").slice(-2, -1)[0]
                  : ""}
              </option>
            ))}
          </select>
        </label>
        <span>
          {t(
            visibleFlow.edges.some((edge) => edge.residual)
              ? "Blue arrows carry activations. Gold arrows carry the residual stream."
              : "Blue arrows carry activations.",
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
          <FitComputation flow={visibleFlow} />
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
          {visibleFlow.nodes.map((node) => (
            <li key={node.id}>
              <strong>{operationName(node)}</strong>
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
                    visibleFlow.edges
                      .filter((edge) => edge.source === node.id)
                      .map((edge) =>
                        operationName(
                          visibleFlow.nodes.find(
                            (item) => item.id === edge.target,
                          )!,
                        ),
                      )
                      .join(" · ") ||
                    t(
                      node.boundary === "output"
                        ? "Outside this path"
                        : "Next block",
                    ),
                })}
              </small>
            </li>
          ))}
        </ol>
        {section !== "all" && (
          <p>
            {t(
              "Boundary nodes show values entering or leaving the selected path.",
            )}
          </p>
        )}
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
