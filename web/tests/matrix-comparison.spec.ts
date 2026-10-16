import { expect, test, type APIRequestContext } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Experiment, MatrixMetrics, RuntimeState } from "../src/lib/types";
import { capturePage } from "./helpers";

type MatrixExperiment = Experiment & { kind: "matrix"; result: MatrixMetrics };
let reference: MatrixExperiment;
let selected: MatrixExperiment;
let otherWeight: MatrixExperiment;
let traceRecord: Experiment;
let before: RuntimeState;
const created: string[] = [];

async function load(request: APIRequestContext, precision: string) {
  const response = await request.post("/api/models/load", {
    data: {
      model_id: before.metadata!.id,
      revision: before.metadata!.revision,
      device: before.device,
      precision,
      context_length: before.context_length,
    },
  });
  expect(response.status()).toBe(202);
  const job = await response.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + job.id)).json()).status,
    )
    .toBe("ready");
}

async function saveMatrix(
  request: APIRequestContext,
  weight: string,
  name: string,
) {
  const response = await request.get(
    "/api/matrix?name=" + encodeURIComponent(weight),
  );
  expect(response.ok()).toBe(true);
  const metrics = await response.json();
  const saving = await request.post("/api/experiments/from-matrix", {
    data: {
      name,
      analysis_id: metrics.analysis_id,
      expected_execution_id: metrics.execution_id,
    },
  });
  expect(saving.status()).toBe(201);
  const record = await saving.json();
  created.push(record.id);
  return (await (
    await request.get("/api/experiments/" + record.id)
  ).json()) as MatrixExperiment;
}

test.beforeAll(async ({ request }) => {
  before = await (await request.get("/api/runtime")).json();
  await load(request, "float32");
  reference = await saveMatrix(
    request,
    "transformer.h.0.attn.c_proj.weight",
    "FP32 projection reference",
  );
  otherWeight = await saveMatrix(
    request,
    "transformer.h.1.attn.c_proj.weight",
    "Unrelated projection",
  );
  const trace = await (
    await request.post("/api/trace", {
      data: { prompt: "the capital of France", max_new_tokens: 0 },
    })
  ).json();
  const saved = await request.post("/api/experiments", {
    data: { name: "Nonmatrix comparison evidence", trace },
  });
  expect(saved.status()).toBe(201);
  traceRecord = await saved.json();
  created.push(traceRecord.id);
  await load(request, "int4");
  selected = await saveMatrix(
    request,
    "transformer.h.0.attn.c_proj.weight",
    "NF4 projection selected",
  );
  expect((await request.delete("/api/runtime")).status()).toBe(204);
});

test.afterAll(async ({ request }) => {
  for (const id of created) await request.delete("/api/experiments/" + id);
  if (before) await load(request, before.precision ?? "auto");
});

async function exported(page: import("@playwright/test").Page) {
  const downloading = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export comparison", exact: true })
    .click();
  const stream = await (await downloading).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString());
}

test("saved matrices compare actual FP32 and NF4 measurements without a loaded model", async ({
  page,
  request,
}) => {
  expect((await (await request.get("/api/runtime")).json()).loaded).toBe(false);
  expect(reference.result.execution_precision).toBe("float32");
  expect(selected.result.execution_precision).toBe("int4");
  expect(reference.result.execution_id).not.toBe(selected.result.execution_id);
  let inference = 0;
  page.on("request", (request) => {
    if (
      /\/api\/(matrix|trace|causal|head-sweep|models\/load)(\?|$)/.test(
        request.url(),
      )
    )
      inference++;
  });
  await page.goto("/experiments?experiment=" + selected.id);
  const picker = page.getByRole("combobox", {
    name: "Reference matrix",
    exact: true,
  });
  await expect(
    picker.locator('option[value="' + traceRecord.id + '"]'),
  ).toHaveCount(0);
  await picker.selectOption(reference.id);
  await expect(
    page.getByRole("status").filter({ hasText: "logical dimensions match" }),
  ).toBeVisible();
  await expect(
    page.getByText("No model loaded", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".matrix-comparison-source").first()).toContainText(
    "NF4",
  );
  await expect(page.locator(".matrix-comparison-source").last()).toContainText(
    "FP32",
  );
  await expect(
    page
      .locator(".recharts-legend-wrapper")
      .getByText("Reference matrix", { exact: true }),
  ).toHaveCount(2);
  await page.getByText("Inspect property changes", { exact: true }).click();
  await expect(page.locator(".matrix-comparison-table tbody tr")).toHaveCount(
    16,
  );
  await expect(
    page
      .locator(".matrix-comparison-table")
      .getByRole("row", { name: /^Exact zeros/ }),
  ).toContainText(" pp");
  const comparison = await exported(page);
  expect(comparison.format).toBe("circuit-lantern-matrix-comparison-v1");
  expect(comparison.selected).toEqual(selected);
  expect(comparison.reference).toEqual(reference);
  expect(comparison.properties).toHaveLength(16);
  for (const row of comparison.properties) {
    expect(row.difference).toBe(
      row.selected === null || row.reference === null
        ? null
        : row.selected - row.reference,
    );
    expect(row.selected_estimated).toBe(false);
    expect(row.reference_estimated).toBe(false);
  }
  expect(
    comparison.properties.some(
      (row: { difference: number | null }) =>
        row.difference !== null && row.difference !== 0,
    ),
  ).toBe(true);
  for (const row of comparison.spectrum) {
    expect(row.selected).toBe(selected.result.singular_values[row.index - 1]);
    expect(row.reference).toBe(reference.result.singular_values[row.index - 1]);
    expect(row.selected_energy).toBe(
      selected.result.cumulative_energy[row.index - 1],
    );
    expect(row.reference_energy).toBe(
      reference.result.cumulative_energy[row.index - 1],
    );
  }
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await capturePage(page, {
    path: "../artifacts/screenshots/saved-matrix-comparison.png",
  });
  const bookmark = page.url();
  await page.reload();
  await expect(picker).toHaveValue(reference.id);
  await expect(
    page.getByRole("button", { name: "Export comparison", exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByText("Inspect property changes", { exact: true }).click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/saved-matrix-comparison-mobile.png",
  });
  await picker.selectOption("");
  await expect(
    page.getByRole("button", { name: "Export comparison", exact: true }),
  ).toHaveCount(0);
  await page.goBack();
  await expect(page).toHaveURL(bookmark);
  await expect(
    page.getByRole("button", { name: "Export comparison", exact: true }),
  ).toBeVisible();
  expect(inference).toBe(0);
  expect(
    await (await request.get("/api/experiments/" + selected.id)).json(),
  ).toEqual(selected);
  expect(
    await (await request.get("/api/experiments/" + reference.id)).json(),
  ).toEqual(reference);
});

for (const [kind, problem] of [
  ["weight", "different weights"],
  ["trace", "not a matrix analysis"],
  ["self", "Choose a different saved matrix"],
] as const) {
  test(
    "an incompatible " + kind + " reference preserves the selected matrix",
    async ({ page }) => {
      const id =
        kind === "weight"
          ? otherWeight.id
          : kind === "trace"
            ? traceRecord.id
            : selected.id;
      await page.goto(
        "/experiments?experiment=" + selected.id + "&compare=" + id,
      );
      await expect(page.getByRole("main").getByRole("alert")).toContainText(
        problem,
      );
      await expect(
        page.getByRole("heading", { name: "Matrix properties", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Export comparison", exact: true }),
      ).toHaveCount(0);
    },
  );
}

for (const [name, patch, problem] of [
  ["repository", { model_id: "other/model" }, "different model repositories"],
  [
    "dimensions",
    { shape: [33, 32], spectrum_shape: [33, 32], elements: 1056 },
    "dimensions differ",
  ],
  ["revision", { revision: "" }, "missing finite matrix measurements"],
  [
    "pooled spectrum",
    { spectrum_shape: [8, 8] },
    "spectrum of the full logical matrix",
  ],
  [
    "missing scalar",
    { frobenius_norm: null },
    "missing finite matrix measurements",
  ],
  [
    "invalid energy",
    { cumulative_energy: [-1] },
    "missing finite matrix measurements",
  ],
] as const) {
  test(
    "a reference with incompatible " +
      name +
      " stays readable without a misleading comparison",
    async ({ page }) => {
      await page.route("**/api/experiments/" + reference.id, async (route) => {
        const response = await route.fetch();
        const record = await response.json();
        await route.fulfill({
          response,
          json: { ...record, result: { ...record.result, ...patch } },
        });
      });
      await page.goto(
        "/experiments?experiment=" + selected.id + "&compare=" + reference.id,
      );
      await expect(page.getByRole("main").getByRole("alert")).toContainText(
        problem,
      );
      await expect(
        page.getByRole("heading", { name: "Matrix properties", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("heading", {
          name: "Singular value comparison",
          exact: true,
        }),
      ).toHaveCount(0);
    },
  );
}

test("estimated spectra retain uncertainty and missing tails in displayed and exported comparisons", async ({
  page,
}) => {
  await page.route("**/api/experiments/" + reference.id, async (route) => {
    const response = await route.fetch();
    const record = await response.json();
    await route.fulfill({
      response,
      json: {
        ...record,
        result: {
          ...record.result,
          spectrum_exact: false,
          spectrum_method: "Full matrix randomized SVD",
          singular_values: record.result.singular_values.slice(0, 8),
          cumulative_energy: record.result.cumulative_energy.slice(0, 8),
        },
      },
    });
  });
  await page.goto(
    "/experiments?experiment=" + selected.id + "&compare=" + reference.id,
  );
  await expect(
    page.getByText(/At least one spectrum is estimated/),
  ).toBeVisible();
  await page.getByText("Inspect property changes", { exact: true }).click();
  const spectral = page
    .locator(".matrix-comparison-table")
    .getByRole("row", { name: /^Spectral norm/ });
  await expect(spectral.getByRole("cell").nth(1)).toContainText("~");
  await expect(spectral.getByRole("cell").nth(2)).toContainText("~");
  const nuclear = page
    .locator(".matrix-comparison-table")
    .getByRole("row", { name: /^Nuclear norm/ });
  await expect(nuclear.getByRole("cell").nth(1)).toHaveText("N/A");
  await expect(nuclear.getByRole("cell").nth(2)).toHaveText("N/A");
  const comparison = await exported(page);
  expect(comparison.reference.result.spectrum_exact).toBe(false);
  expect(comparison.spectrum).toHaveLength(
    selected.result.singular_values.length,
  );
  expect(comparison.spectrum[8]).toMatchObject({
    reference: null,
    reference_energy: null,
    difference: null,
  });
  expect(
    comparison.properties.find(
      (row: { key: string }) => row.key === "spectral_norm",
    ),
  ).toMatchObject({
    selected_estimated: false,
    reference_estimated: true,
    difference_estimated: true,
  });
  expect(
    comparison.properties.find(
      (row: { key: string }) => row.key === "nuclear_norm",
    ),
  ).toMatchObject({
    reference: null,
    difference: null,
  });
});

test("a missing reference leaves the primary saved matrix available", async ({
  page,
}) => {
  await page.goto(
    "/experiments?experiment=" +
      selected.id +
      "&compare=00000000000000000000000000000000",
  );
  await expect(page.getByRole("main").getByRole("alert")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Matrix properties", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Export comparison", exact: true }),
  ).toHaveCount(0);
});
