import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

test.beforeAll(async ({ request }) => {
  const runtime = await (await request.get("/api/runtime")).json();
  expect(runtime.precision).toBe("float32");
});

test("a selected prompt query is captured, exported and reopened with its routing labels", async ({
  page,
  request,
}) => {
  await page.goto("/playground");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France");
  await page.getByLabel("Generated tokens").selectOption("0");
  await page.getByText("Attention capture options", { exact: true }).click();
  const query = page.getByRole("combobox", {
    name: "Attention routing query",
    exact: true,
  });
  await query.selectOption("1");
  await expect(
    page.getByRole("button", { name: "Run & trace", exact: true }),
  ).toBeEnabled();
  const capturing = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  const response = await capturing;
  expect(response.request().postDataJSON().attention_query_position).toBe(1);
  const trace = await response.json();
  expect(trace.attention_query_position).toBe(1);
  expect(trace.attention.length).toBeGreaterThan(0);
  for (const layer of trace.attention) {
    expect(layer.query_position).toBe(1);
    for (const row of layer.query_by_head) {
      expect(row).toHaveLength(trace.token_ids.length);
      expect(row.slice(2).every((value: number) => value === 0)).toBe(true);
      expect(
        row.reduce((sum: number, value: number) => sum + value, 0),
      ).toBeCloseTo(1, 5);
    }
  }
  await page.getByRole("tab", { name: "Attention", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Prompt token attention by head",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText("Layer 0 · Query position 1 · capital", { exact: true }),
  ).toBeVisible();
  await page.getByText("Inspect head routing values", { exact: true }).click();
  await expect(
    page.getByRole("columnheader", { name: "QUERY ENTROPY", exact: true }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/attention-selected-query.png",
    fullPage: true,
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export trace", exact: true }).click();
  const stream = await (await downloading).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  expect(JSON.parse(Buffer.concat(chunks).toString())).toEqual(trace);
  await page
    .getByLabel("Experiment name")
    .fill("Selected routing query browser capture");
  const saving = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/experiments") &&
      response.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Save experiment", exact: true })
    .click();
  const record = await (await saving).json();
  try {
    await expect(
      page.getByRole("button", { name: "Saved to experiments", exact: true }),
    ).toBeVisible();
    await page.goto("/experiments?experiment=" + record.id);
    await page.getByRole("tab", { name: "Attention", exact: true }).click();
    await expect(
      page.getByText("Layer 0 · Query position 1 · capital", { exact: true }),
    ).toBeVisible();
    await page.getByLabel("Attention layer").selectOption("2");
    await expect(
      page.getByText("Layer 2 · Query position 1 · capital", { exact: true }),
    ).toBeVisible();
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await capturePage(page, {
      path: "../artifacts/screenshots/attention-selected-query-notebook-mobile.png",
      fullPage: true,
    });
  } finally {
    await request.delete("/api/experiments/" + record.id);
  }
});

test("changing the prompt resets a selected query to the final token", async ({
  page,
}) => {
  await page.goto("/playground");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France");
  await page.getByLabel("Generated tokens").selectOption("0");
  await page.getByText("Attention capture options", { exact: true }).click();
  const query = page.getByRole("combobox", {
    name: "Attention routing query",
    exact: true,
  });
  await query.selectOption("2");
  await page.getByLabel("Prompt", { exact: true }).fill("the capital");
  await expect(query).toHaveValue("-1");
  const capturing = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  const trace = await (await capturing).json();
  expect(trace.attention_query_position).toBe(trace.token_ids.length - 1);
  await page.getByRole("tab", { name: "Attention", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Final token attention by head",
      exact: true,
    }),
  ).toBeVisible();
  await page.getByLabel("Capture attention", { exact: true }).uncheck();
  await expect(query).toHaveCount(0);
  const without = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  const noAttention = await (await without).json();
  expect(noAttention.attention).toEqual([]);
  expect(noAttention.attention_query_position).toBeNull();
});

test("the API rejects an out of prompt query before another capture can run", async ({
  request,
}) => {
  const invalid = await request.post("/api/trace", {
    data: {
      prompt: "the capital",
      max_new_tokens: 0,
      capture_attention: true,
      attention_query_position: 99,
    },
  });
  expect(invalid.status()).toBe(400);
  expect((await invalid.json()).detail).toContain(
    "query position inside the prompt",
  );
  const first = await request.post("/api/trace", {
    data: {
      prompt: "the capital",
      max_new_tokens: 0,
      capture_attention: true,
      attention_query_position: 0,
    },
  });
  expect(first.ok()).toBe(true);
  const trace = await first.json();
  expect(trace.attention_query_position).toBe(0);
  for (const layer of trace.attention) {
    expect(
      layer.query_head_entropy.every((value: number) => Math.abs(value) < 1e-6),
    ).toBe(true);
    expect(
      layer.query_by_head.every(
        (row: number[]) =>
          row[0] === 1 && row.slice(1).every((value) => value === 0),
      ),
    ).toBe(true);
  }
});

test("older saved routing rows without a query position retain their final token label", async ({
  page,
  request,
}) => {
  const response = await request.post("/api/trace", {
    data: {
      prompt: "the capital of France",
      max_new_tokens: 0,
      capture_attention: true,
    },
  });
  expect(response.ok()).toBe(true);
  const trace = await response.json();
  delete trace.attention_query_position;
  for (const layer of trace.attention) delete layer.query_position;
  const saving = await request.post("/api/experiments", {
    data: { name: "Legacy final routing query", trace },
  });
  expect(saving.status()).toBe(201);
  const record = await saving.json();
  try {
    await page.goto("/experiments?experiment=" + record.id);
    await page.getByRole("tab", { name: "Attention", exact: true }).click();
    await expect(
      page.getByRole("heading", {
        name: "Final token attention by head",
        exact: true,
      }),
    ).toBeVisible();
    await expect(
      page.getByText("Layer 0 · Query position 3 · France", { exact: true }),
    ).toBeVisible();
  } finally {
    await request.delete("/api/experiments/" + record.id);
  }
});
