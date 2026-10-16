import { capturePage } from "./helpers";
import { expect, test } from "@playwright/test";

test("a researcher can focus a block and inspect a matrix directly", async ({
  page,
  request,
}) => {
  const runtime = await (await request.get("/api/runtime")).json();
  test.skip(
    !runtime.loaded,
    "A model is required to inspect its architecture.",
  );
  await page.goto("/workspace");
  await page.getByLabel("Focus on transformer block").selectOption("0");
  await expect(
    page.getByText("Block 0 matrices", { exact: true }),
  ).toBeVisible();
  const graph = page.locator(".graph-container");
  const matrix = graph
    .getByRole("button", { name: /Inspect .*?(q_proj|c_attn).*?weight/ })
    .first();
  await expect(matrix).toBeVisible();
  await page.waitForTimeout(300);
  await capturePage(page, {
    path: "../artifacts/screenshots/graph-block-focus.png",
    fullPage: true,
  });
  await matrix.click();
  await expect(
    page.getByText("Singular value spectrum", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Focus on transformer block").selectOption("all");
  await expect(
    page.getByText("All weight matrices", { exact: true }),
  ).toBeVisible();
});

test("tied output projection opens the shared embedding matrix", async ({
  page,
  request,
}) => {
  const graph = await (await request.get("/api/graph")).json();
  expect(graph.metadata.tied_embeddings).toBe(true);
  const input = graph.matrices.find((weight: { name: string }) =>
    /(?:wte|embed_tokens|embed_in)\.weight$/.test(weight.name),
  );
  await page.goto("/workspace");
  const output = page.locator('.react-flow__node[data-id="unembedding"]');
  await expect(output.getByText("LOGITS · SHARED W_E")).toBeVisible();
  const projection = output.getByRole("button", {
    name: /shared output projection/,
  });
  await expect(projection).toContainText("W_U");
  await projection.click();
  await expect(
    page.getByRole("heading", { name: input.name, exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Singular value spectrum", { exact: true }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/shared-output-projection.png",
  });
});

test("architecture framing follows container resizing while retaining the focused block", async ({
  page,
}) => {
  await page.goto("/workspace");
  const graph = page.locator(".graph-container");
  const embedding = graph.locator('.react-flow__node[data-id="embedding"]');
  await expect(embedding).toBeVisible();
  async function expectCentered(node: import("@playwright/test").Locator) {
    await expect
      .poll(async () => {
        const container = await graph.boundingBox();
        const box = await node.boundingBox();
        if (!container || !box) return Number.POSITIVE_INFINITY;
        return Math.abs(
          box.x + box.width / 2 - container.x - container.width / 2,
        );
      })
      .toBeLessThan(4);
  }
  await expectCentered(embedding);
  await page.setViewportSize({ width: 390, height: 844 });
  await expectCentered(embedding);
  await capturePage(page, {
    path: "../artifacts/screenshots/graph-responsive-mobile.png",
  });
  await page.getByLabel("Focus on transformer block").selectOption("0");
  const block = graph.locator('.react-flow__node[data-id="0"]');
  await expectCentered(block);
  await expect
    .poll(async () => {
      const container = await graph.boundingBox();
      const box = await block.boundingBox();
      if (!container || !box) return Number.POSITIVE_INFINITY;
      return Math.abs(
        box.y + box.height / 2 - container.y - container.height / 2,
      );
    })
    .toBeLessThan(4);
  const count = await graph.locator(".react-flow__node").count();
  await block.focus();
  await page.keyboard.press("Delete");
  await expect(graph.locator(".react-flow__node")).toHaveCount(count);
  await page.setViewportSize({ width: 1024, height: 768 });
  await expect(page.getByLabel("Focus on transformer block")).toHaveValue("0");
  await expectCentered(block);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/graph-responsive-tablet.png",
  });
});

test("blocks with different matrix counts use measured heights without overlap", async ({
  page,
}) => {
  await page.route("**/api/graph**", async (route) => {
    const response = await route.fetch();
    const graph = await response.json();
    // Extra operators exercise taller graph cards without another checkpoint.
    graph.matrices.push(
      ...graph.matrices
        .filter((weight: { name: string }) => weight.name.includes(".h.0."))
        .map((weight: { name: string }) => ({
          ...weight,
          name: weight.name.replace(".h.0.", ".h.0.extra."),
        })),
    );
    await route.fulfill({ response, json: graph });
  });
  await page.goto("/workspace");
  const graph = page.locator(".graph-container");
  const first = graph.locator('.react-flow__node[data-id="0"]');
  const second = graph.locator('.react-flow__node[data-id="1"]');
  await page.getByLabel("Focus on transformer block").selectOption("0");
  async function expectGap() {
    await expect
      .poll(async () => {
        const a = await first.boundingBox();
        const b = await second.boundingBox();
        return a && b ? b.y - a.y - a.height : Number.NEGATIVE_INFINITY;
      })
      .toBeGreaterThan(2);
  }
  await expectGap();
  await capturePage(page, {
    path: "../artifacts/screenshots/graph-variable-height.png",
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expectGap();
  await capturePage(page, {
    path: "../artifacts/screenshots/graph-variable-height-mobile.png",
  });
});
