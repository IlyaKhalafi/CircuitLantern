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
