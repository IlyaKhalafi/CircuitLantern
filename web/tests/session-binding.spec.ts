import { expect, test, type APIRequestContext } from "@playwright/test";
import { capturePage } from "./helpers";

async function loadModel(
  request: APIRequestContext,
  precision: "float32" | "int4",
) {
  const response = await request.post("/api/models/load", {
    data: {
      model_id: "hf-internal-testing/tiny-random-gpt2",
      precision,
      context_length: 256,
    },
  });
  expect(response.status(), await response.text()).toBe(202);
  const job = await response.json();
  await expect
    .poll(async () => {
      const current = await (await request.get(`/api/jobs/${job.id}`)).json();
      if (current.status === "failed") throw new Error(current.message);
      return current.status;
    })
    .toBe("ready");
  return (await request.get("/api/runtime")).json();
}

test("an old graph cannot analyze replacement weights and exports the refreshed source", async ({
  page,
  request,
}) => {
  const original = await loadModel(request, "float32");
  await page.goto("/workspace");
  await expect(
    page.getByText("All weight matrices", { exact: true }),
  ).toBeVisible();
  const replacement = await loadModel(request, "int4");
  expect(replacement.metadata.execution_id).not.toBe(
    original.metadata.execution_id,
  );
  const inspect = async () => {
    await page.getByLabel("Filter matrices").fill("h.0.attn.c_attn");
    await page
      .locator(".weight-list")
      .getByRole("button", { name: /transformer.h.0.attn.c_attn.weight/ })
      .click();
  };
  await inspect();
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "active model changed",
  );
  await expect(
    page.getByText("Singular value spectrum", { exact: true }),
  ).toHaveCount(0);
  await capturePage(page, {
    path: "../artifacts/screenshots/stale-model-explorer.png",
  });
  await page.getByRole("button", { name: "Refresh explorer" }).click();
  await expect(
    page.getByText("All weight matrices", { exact: true }),
  ).toBeVisible();
  await inspect();
  await expect(
    page.getByText(/Statistics describe the dequantized float32 matrix/),
  ).toBeVisible();
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export analysis" }).click();
  const stream = await (await downloading).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const analysis = JSON.parse(Buffer.concat(chunks).toString());
  expect(analysis.execution_id).toBe(replacement.metadata.execution_id);
  expect(analysis.model_id).toBe(replacement.metadata.id);
  expect(analysis.revision).toBe(replacement.metadata.revision);
  expect(analysis.execution_precision).toBe("int4");
  expect(analysis.shape).toEqual([96, 32]);
});

for (const precision of ["int4", "float32"] as const) {
  test(`activation refresh preserves drafts and isolates ${precision} session baselines`, async ({
    page,
    request,
  }) => {
    await loadModel(request, "float32");
    await page.goto("/playground");
    await page
      .getByLabel("Prompt", { exact: true })
      .fill("the capital of France");
    await page.getByLabel("Generated tokens").selectOption("4");
    await page.getByRole("button", { name: "Run & trace" }).click();
    await expect(
      page.getByText("Capture complete", { exact: true }),
    ).toBeVisible();
    const replacement = await loadModel(request, precision);
    await page.getByRole("button", { name: "Run & trace" }).click();
    await expect(page.getByRole("main").getByRole("alert")).toContainText(
      "active model changed",
    );
    await page.getByRole("button", { name: "Refresh runtime" }).click();
    await expect(
      page.getByText(/This capture belongs to a previous model session/),
    ).toBeVisible();
    await expect(page.getByLabel("Prompt", { exact: true })).toHaveValue(
      "the capital of France",
    );
    await page.getByLabel("Layer intervention").selectOption("0");
    const capturing = page.waitForResponse((response) =>
      response.url().endsWith("/api/trace"),
    );
    await page.getByRole("button", { name: "Run & trace" }).click();
    const trace = await (await capturing).json();
    expect(trace.execution_id).toBe(replacement.metadata.execution_id);
    expect(trace.ablate_layer).toBe(0);
    await expect(page.locator(".recharts-legend-wrapper")).toHaveCount(0);
    await expect(
      page
        .locator(".trace-meta")
        .getByText(precision === "int4" ? "NF4 · 4 bit" : "FP32", {
          exact: true,
        }),
    ).toBeVisible();
    await expect(
      page.getByText(/This capture belongs to a previous model session/),
    ).toHaveCount(0);
    await capturePage(page, {
      path: `../artifacts/screenshots/refreshed-model-capture-${precision}.png`,
    });
  });
}

test("causal submissions require the displayed session and recover after refreshing", async ({
  page,
  request,
}) => {
  await loadModel(request, "float32");
  await page.goto("/causal");
  await expect(page.getByLabel("Clean prompt")).toBeVisible();
  const replacement = await loadModel(request, "int4");
  await page.getByRole("button", { name: "Run patch sweep" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "active model changed",
  );
  await page.getByRole("button", { name: "Refresh runtime" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  await expect(page.getByLabel("Clean prompt")).toBeVisible();
  await page.getByRole("button", { name: "Run patch sweep" }).click();
  await expect(page.getByText("Sweep complete", { exact: true })).toBeVisible();
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export sweep" }).click();
  const stream = await (await downloading).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  expect(JSON.parse(Buffer.concat(chunks).toString()).execution_id).toBe(
    replacement.metadata.execution_id,
  );
});

test("a stale server page cannot unload a replacement model", async ({
  page,
  request,
}) => {
  await loadModel(request, "float32");
  try {
    await page.goto("/hardware");
    await expect(
      page.getByRole("button", { name: "Unload model" }),
    ).toBeVisible();
    const replacement = await loadModel(request, "int4");
    await page.getByRole("button", { name: "Unload model" }).click();
    await expect(page.getByRole("main").getByRole("alert")).toContainText(
      "active model changed",
    );
    expect(
      (await (await request.get("/api/runtime")).json()).metadata.execution_id,
    ).toBe(replacement.metadata.execution_id);
    await page.getByRole("button", { name: "Refresh snapshot" }).click();
    await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
    await page.getByRole("button", { name: "Unload model" }).click();
    await expect
      .poll(
        async () => (await (await request.get("/api/runtime")).json()).loaded,
      )
      .toBe(false);
  } finally {
    await loadModel(request, "float32");
  }
});
