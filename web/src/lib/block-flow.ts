import type { BlockFlow } from "./types";

const mlpNodes = new Set([
  "norm-mlp",
  "up",
  "gate",
  "act",
  "multiply",
  "down",
  "output",
]);

export function hasFlowSection(flow: BlockFlow, section: string) {
  return (
    section === "all" ||
    ((section === "attention" || section === "mlp") &&
      (!flow.provenance || flow.nodes.some((node) => node.section === section)))
  );
}

/** Boundaries represent tensor values, never unshown consumer operations. */
export function focusBlockFlow(flow: BlockFlow, section: string): BlockFlow {
  if (section === "all" || !flow.supported || !hasFlowSection(flow, section))
    return flow;
  const core = flow.nodes.filter((node) =>
    flow.provenance
      ? node.section === section
      : section === "attention"
        ? !mlpNodes.has(node.id)
        : mlpNodes.has(node.id),
  );
  if (!core.length) return flow;
  const ids = new Set(core.map((node) => node.id));
  const entries = new Map<string, BlockFlow["nodes"][number]>();
  const exits = new Map<string, BlockFlow["nodes"][number]>();
  const nodesById = new Map(flow.nodes.map((node) => [node.id, node]));
  const edges: BlockFlow["edges"] = [];
  for (const edge of flow.edges) {
    if (ids.has(edge.source) && ids.has(edge.target)) edges.push(edge);
    else if (!ids.has(edge.source) && ids.has(edge.target)) {
      const origin = nodesById.get(edge.source);
      if (!origin) continue;
      const id = "boundary-in-" + origin.id;
      if (!entries.has(id))
        entries.set(id, {
          id,
          label: "Path input",
          detail: origin.label,
          weight: null,
          x: 0,
          y: 0,
          boundary: "input",
          boundary_origin: origin.id,
        });
      edges.push({ ...edge, source: id });
    } else if (ids.has(edge.source) && !ids.has(edge.target)) {
      // Several external consumers share this same produced value. Draw one
      // endpoint without pretending the external operations are inside the path.
      const origin = nodesById.get(edge.source)!;
      const id = "boundary-out-" + origin.id;
      if (!exits.has(id)) {
        exits.set(id, {
          id,
          label: "Path output",
          detail: origin.label,
          weight: null,
          x: 0,
          y: 0,
          boundary: "output",
          boundary_origin: origin.id,
        });
        edges.push({
          source: origin.id,
          target: id,
          label: "",
          residual: false,
        });
      }
    }
  }
  const top = Math.min(...core.map((node) => node.y)) - 115;
  const bottom = Math.max(...core.map((node) => node.y)) + 115;
  const place = (values: Map<string, BlockFlow["nodes"][number]>, y: number) =>
    [...values.values()].map((node, index, group) => ({
      ...node,
      x: 240 + (index - (group.length - 1) / 2) * 230,
      y,
    }));
  return {
    ...flow,
    nodes: [...place(entries, top), ...core, ...place(exits, bottom)],
    edges,
  };
}
