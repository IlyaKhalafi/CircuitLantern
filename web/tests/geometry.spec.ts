import { capturePage } from "./helpers";
import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("representation geometry is inspectable, exportable and saved with experiments", async ({
  page,
}) => {
  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("8");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Representation geometry" }).click();
  await expect(
    page.getByText("Layer representation similarity", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Token representation similarity", { exact: true }),
  ).toBeVisible();
  await page.getByText("Inspect block update values", { exact: true }).click();
  await expect(
    page.getByRole("columnheader", { name: "UPDATE / INPUT NORM" }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/representation-geometry.png",
    fullPage: true,
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export trace" }).click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const trace = JSON.parse(Buffer.concat(chunks).toString());
  expect(trace.geometry.cka).toHaveLength(trace.layers.length);
  expect(trace.geometry.token_cosine).toHaveLength(trace.tokens.length);
  expect(trace.layers[1].update_norm).toBeGreaterThan(0);
  await page
    .getByLabel("Experiment name")
    .fill("Representation geometry verification");
  await page.getByRole("button", { name: "Save experiment" }).click();
  await expect(
    page.getByRole("button", { name: "Saved to experiments" }),
  ).toBeVisible();
  await page.goto("/experiments");
  await page
    .getByRole("link", { name: /Representation geometry verification/ })
    .click();
  await page.getByRole("tab", { name: "Representation geometry" }).click();
  await expect(
    page.getByText("Residual updates", { exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  const activeTab = await page
    .getByRole("tab", { name: "Representation geometry" })
    .boundingBox();
  expect(activeTab!.x).toBeGreaterThanOrEqual(0);
  expect(activeTab!.x + activeTab!.width).toBeLessThanOrEqual(390);
  await capturePage(page, {
    path: "../artifacts/screenshots/representation-geometry-mobile.png",
    fullPage: true,
  });
});

test("one token gives a useful explanation instead of a false CKA score", async ({
  page,
}) => {
  await page.goto("/playground");
  await page.getByLabel("Prompt", { exact: true }).fill("the");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Representation geometry" }).click();
  await expect(page.getByText(/Use at least 3 prompt tokens/)).toBeVisible();
  await expect(
    page.getByText("Token representation similarity", { exact: true }),
  ).toBeVisible();
});

test("older notebook traces remain readable without representation geometry", async ({
  page,
  request,
}) => {
  const capture = await request.post("/api/trace", {
    data: {
      prompt: "the capital of France is",
      max_new_tokens: 1,
      capture_attention: false,
      ablate_layer: null,
    },
  });
  expect(capture.ok()).toBe(true);
  const trace = await capture.json();
  delete trace.geometry;
  for (const layer of trace.layers) {
    delete layer.update_norm;
    delete layer.update_rms;
    delete layer.relative_update;
  }
  const saved = await request.post("/api/experiments", {
    data: {
      name: "Legacy trace verification",
      notes: "",
      trace,
    },
  });
  expect(saved.ok()).toBe(true);
  await page.goto("/experiments");
  await page.getByRole("link", { name: /Legacy trace verification/ }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("tab", { name: "Representation geometry" }),
  ).toHaveCount(0);
  await expect(
    page.getByText("Residual stream magnitude", { exact: true }),
  ).toBeVisible();
});
