import { expect, test } from "@playwright/test";
import { focusBlockFlow } from "../src/lib/block-flow";
import type { BlockFlow } from "../src/lib/types";
import { capturePage } from "./helpers";

async function hybrid(page: import("@playwright/test").Page) {
  const graph = await (
    await page.request.get("/api/test/hybrid-graph?layers=12")
  ).json();
  await page.route("**/api/graph**", (route) => route.fulfill({ json: graph }));
  return graph;
}

for (const section of ["attention", "mlp"]) {
  test(`focused ${section} preserves real dependencies and explicit tensor boundaries`, async ({
    request,
  }) => {
    const graph = await (await request.get("/api/test/hybrid-graph")).json();
    for (const flow of graph.dataflow as BlockFlow[]) {
      const original = JSON.stringify(flow);
      const focused = focusBlockFlow(flow, section);
      const ids = new Set(focused.nodes.map((node) => node.id));
      const core = flow.nodes.filter((node) => node.section === section);
      expect(focused.nodes.filter((node) => !node.boundary)).toEqual(core);
      expect(
        focused.nodes.filter((node) => node.weight).map((node) => node.weight),
      ).toEqual(core.filter((node) => node.weight).map((node) => node.weight));
      expect(
        focused.nodes.filter((node) => node.boundary === "input"),
      ).toHaveLength(1);
      expect(
        focused.nodes.filter((node) => node.boundary === "output"),
      ).toHaveLength(1);
      expect(
        focused.edges.every(
          (edge) => ids.has(edge.source) && ids.has(edge.target),
        ),
      ).toBe(true);
      for (const boundary of focused.nodes.filter((node) => node.boundary)) {
        expect(boundary.weight).toBeNull();
        expect(
          flow.nodes.some((node) => node.id === boundary.boundary_origin),
        ).toBe(true);
        if (boundary.boundary === "output") {
          const incoming = focused.edges.filter(
            (edge) => edge.target === boundary.id,
          );
          expect(incoming).toHaveLength(1);
          expect(incoming[0].source).toBe(boundary.boundary_origin);
        }
      }
      expect(JSON.stringify(flow)).toBe(original);
      expect(focusBlockFlow(flow, "all")).toBe(flow);
    }
  });
}

test("curated MLP paths preserve their real residual input and unsupported sections remain unchanged", async ({
  request,
}) => {
  const graph = await (await request.get("/api/graph")).json();
  const flow = graph.dataflow[0] as BlockFlow;
  const focused = focusBlockFlow(flow, "mlp");
  const boundary = focused.nodes.find((node) => node.boundary === "input")!;
  expect(boundary.boundary_origin).toBe("add-attn");
  expect(focused.edges).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ source: boundary.id, target: "norm-mlp" }),
      expect.objectContaining({
        source: boundary.id,
        target: "output",
        residual: true,
      }),
    ]),
  );
  // This terminal node already represents the value leaving the whole block.
  // It must not acquire a duplicate synthetic endpoint or a missing consumer.
  expect(focused.nodes.find((node) => node.id === "output")).toEqual(
    flow.nodes.find((node) => node.id === "output"),
  );
  expect(focused.edges.some((edge) => edge.source === "output")).toBe(false);
  const unsupported: BlockFlow = {
    layer: 0,
    supported: false,
    nodes: [],
    edges: [],
  };
  expect(focusBlockFlow(unsupported, "attention")).toBe(unsupported);
});

test("layer arrows and keyboard navigation reveal the complete layer strip", async ({
  page,
}) => {
  await hybrid(page);
  await page.goto("/workspace");
  await expect(
    page.getByRole("button", { name: "Previous block", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Next block", exact: true }).click();
  await expect(page.getByLabel("Focus on transformer block")).toHaveValue("1");
  const strip = page.locator(".flow-layer-pattern");
  await strip.locator('[aria-pressed="true"]').focus();
  await page.keyboard.press("End");
  await expect(page.getByLabel("Focus on transformer block")).toHaveValue("11");
  await expect(strip.locator('[aria-pressed="true"]')).toBeFocused();
  await expect(
    page.getByRole("button", { name: "Next block", exact: true }),
  ).toBeDisabled();
  await expect
    .poll(async () => {
      const parent = (await strip.boundingBox())!,
        active = (await strip.locator('[aria-pressed="true"]').boundingBox())!;
      return (
        active.x >= parent.x &&
        active.x + active.width <= parent.x + parent.width
      );
    })
    .toBe(true);
  await page.keyboard.press("ArrowLeft");
  await expect(page.getByLabel("Focus on transformer block")).toHaveValue("10");
  await page.keyboard.press("Home");
  await expect(page.getByLabel("Focus on transformer block")).toHaveValue("0");
  await expect(strip.locator('button[tabindex="0"]')).toHaveCount(1);
});

test("layer navigation preserves modified shortcuts and shows keyboard focus", async ({
  page,
}) => {
  await hybrid(page);
  await page.goto("/workspace");
  const active = page.locator('.flow-layer-pattern [aria-pressed="true"]');
  const unfocused = await active.evaluate(
    (node) => getComputedStyle(node).outline,
  );
  await active.focus();
  await expect(active).toBeFocused();
  expect(
    await active.evaluate((node) => getComputedStyle(node).outline),
  ).not.toBe(unfocused);
  expect(
    await active.evaluate((node) => getComputedStyle(node).outlineOffset),
  ).toBe("-3px");
  for (const modifier of ["altKey", "ctrlKey", "metaKey", "shiftKey"]) {
    for (const key of ["ArrowRight", "End"]) {
      const allowed = await active.evaluate(
        (node, options) =>
          node.dispatchEvent(
            new KeyboardEvent("keydown", {
              bubbles: true,
              cancelable: true,
              ...options,
            }),
          ),
        { key, [modifier]: true },
      );
      expect(allowed).toBe(true);
      await expect(page.getByLabel("Focus on transformer block")).toHaveValue(
        "0",
      );
    }
  }
});

test("focused views show complete boundary values and preserve the selected path across layers", async ({
  page,
}) => {
  await hybrid(page);
  await page.goto("/workspace");
  await expect(page.locator(".flow-boundary")).toHaveCount(2);
  await page.getByLabel("Diagram focus", { exact: true }).selectOption("mlp");
  await expect(page.locator(".flow-weight")).toHaveCount(3);
  await page.getByRole("button", { name: "Next block", exact: true }).click();
  await expect(page.getByLabel("Diagram focus", { exact: true })).toHaveValue(
    "mlp",
  );
  await expect(page.locator(".flow-weight")).toHaveCount(3);
  await expect
    .poll(async () => {
      const canvas = (await page.locator(".block-dataflow").boundingBox())!;
      const boxes = await page.locator(".flow-boundary").all();
      for (const node of boxes) {
        const box = (await node.boundingBox())!;
        if (box.y < canvas.y || box.y + box.height > canvas.y + canvas.height)
          return false;
      }
      return true;
    })
    .toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/focused-mlp-boundaries.png",
  });
  await page.getByLabel("Diagram focus", { exact: true }).selectOption("all");
  await expect(page.locator(".flow-boundary")).toHaveCount(0);
});

test("changing focused paths clears unrelated connection highlighting", async ({
  page,
}) => {
  await hybrid(page);
  await page.goto("/workspace");
  const select = page.getByLabel("Trace connections", { exact: true });
  await select.selectOption({ label: "Update projection · in_proj_b" });
  await expect(page.locator(".flow-dimmed").first()).toBeVisible();
  await page.getByLabel("Diagram focus", { exact: true }).selectOption("mlp");
  await expect(select).toHaveValue("");
  await expect(page.locator(".flow-dimmed")).toHaveCount(0);
});

for (const locale of ["fa", "ar"]) {
  test(`layer navigation follows physical arrow direction in ${locale}`, async ({
    page,
    context,
  }) => {
    await context.addCookies([
      {
        name: "lantern_locale",
        value: locale,
        url: new URL(test.info().project.use.baseURL as string).origin,
      },
    ]);
    await hybrid(page);
    await page.goto("/workspace");
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
    const strip = page.locator(".flow-layer-pattern");
    await strip.locator('[aria-pressed="true"]').focus();
    await page.keyboard.press("ArrowLeft");
    await expect(page.locator(".graph-toolbar select")).toHaveValue("1");
    await page.keyboard.press("ArrowRight");
    await expect(page.locator(".graph-toolbar select")).toHaveValue("0");
    await page.keyboard.press("End");
    await expect(page.locator(".graph-toolbar select")).toHaveValue("11");
    await page.setViewportSize({ width: 390, height: 844 });
    await expect
      .poll(async () => {
        const parent = (await strip.boundingBox())!,
          active = (await strip
            .locator('[aria-pressed="true"]')
            .boundingBox())!;
        return (
          active.x >= parent.x &&
          active.x + active.width <= parent.x + parent.width
        );
      })
      .toBe(true);
    await capturePage(page, {
      path: `../artifacts/screenshots/focused-layer-navigation-${locale}.png`,
    });
  });
}

test("navigation retains a reachable focus when an end button becomes disabled", async ({
  page,
}) => {
  await hybrid(page);
  await page.goto("/workspace");
  await page.getByLabel("Focus on transformer block").selectOption("10");
  const next = page.getByRole("button", { name: "Next block", exact: true });
  await next.focus();
  await page.keyboard.press("Enter");
  await expect(next).toBeDisabled();
  await expect(
    page.locator('.flow-layer-pattern [aria-pressed="true"]'),
  ).toBeFocused();
  await page.getByLabel("Focus on transformer block").selectOption("1");
  const previous = page.getByRole("button", {
    name: "Previous block",
    exact: true,
  });
  await previous.focus();
  await page.keyboard.press("Enter");
  await expect(previous).toBeDisabled();
  await expect(
    page.locator('.flow-layer-pattern [aria-pressed="true"]'),
  ).toBeFocused();
});

test("parallel residual boundaries retain distinct values and readable origin labels", async ({
  page,
}) => {
  const graph = await (
    await page.request.get("/api/test/parallel-graph")
  ).json();
  const flow = graph.dataflow[0] as BlockFlow;
  const mlp = focusBlockFlow(flow, "mlp");
  expect(
    new Set(
      mlp.nodes
        .filter((node) => node.boundary === "input")
        .map((node) => node.boundary_origin),
    ),
  ).toEqual(new Set(["input", "out"]));
  const attention = focusBlockFlow(flow, "attention");
  expect(
    new Set(
      attention.nodes
        .filter((node) => node.boundary === "output")
        .map((node) => node.boundary_origin),
    ),
  ).toEqual(new Set(["input", "out"]));
  await page.route("**/api/graph**", (route) => route.fulfill({ json: graph }));
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "Block dataflow", exact: true })
    .click();
  await page.getByLabel("Diagram focus", { exact: true }).selectOption("mlp");
  const trace = page.getByLabel("Trace connections", { exact: true });
  await expect(
    trace.locator("option").filter({ hasText: /^Path input · / }),
  ).toHaveCount(2);
  await expect(
    trace.locator("option", { hasText: "Path input · Residual input" }),
  ).toHaveCount(1);
  await expect(
    trace.locator("option", { hasText: "Path input · W_O" }),
  ).toHaveCount(1);
  await page
    .getByLabel("Diagram focus", { exact: true })
    .selectOption("attention");
  await expect(
    trace.locator("option", { hasText: "Path output · Residual input" }),
  ).toHaveCount(1);
  await expect(
    trace.locator("option", { hasText: "Path output · W_O" }),
  ).toHaveCount(1);
});

test("an incomplete layer list keeps a reachable keyboard recovery path", async ({
  page,
}) => {
  const graph = await hybrid(page);
  graph.dataflow = graph.dataflow.filter(
    (flow: { layer: number }) => flow.layer !== 0,
  );
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "Block dataflow", exact: true })
    .click();
  const first = page.locator(".flow-layer-pattern button").first();
  await expect(first).toHaveAttribute("tabindex", "0");
  await first.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Focus on transformer block")).toHaveValue("1");
  await expect(page.getByText("8 of 8 block matrices connected")).toBeVisible();
});
