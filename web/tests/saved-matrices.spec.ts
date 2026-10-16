import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { MatrixMetrics, RuntimeState } from "../src/lib/types";
import { capturePage } from "./helpers";

async function restore(request: APIRequestContext, before: RuntimeState) {
  const response = await request.post("/api/models/load", {
    data: {
      model_id: before.metadata!.id,
      revision: before.metadata!.revision,
      device: before.device,
      precision: before.precision,
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

async function inspect(
  page: Page,
  name = "wte.weight",
): Promise<MatrixMetrics> {
  await page.goto("/workspace");
  await page.getByLabel("Filter matrices").fill(name);
  const capturing = page.waitForResponse(
    (response) => response.url().includes("/api/matrix?") && response.ok(),
  );
  await page.locator(".weight-item").first().click();
  const metrics = await (await capturing).json();
  await expect(
    page.getByRole("button", { name: "Save experiment", exact: true }),
  ).toBeEnabled();
  return metrics;
}

async function exported(page: Page) {
  const downloading = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export analysis", exact: true })
    .click();
  const stream = await (await downloading).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString());
}

test("matrix measurements remain editable and inspectable after a real model unload", async ({
  page,
  request,
}) => {
  const before: RuntimeState = await (await request.get("/api/runtime")).json();
  let recordId: string | undefined;
  let unloaded = false;
  try {
    const metrics = await inspect(page);
    expect(metrics.analysis_id).toMatch(/^[a-f0-9]{32}$/);
    expect(await exported(page)).toEqual(metrics);
    await page.getByLabel("Experiment name").fill("Saved embedding geometry");
    const saving = page.waitForResponse((response) =>
      response.url().endsWith("/api/experiments/from-matrix"),
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    const response = await saving;
    expect(response.status()).toBe(201);
    expect(response.request().postDataJSON()).toMatchObject({
      analysis_id: metrics.analysis_id,
      expected_execution_id: metrics.execution_id,
    });
    const record = await response.json();
    recordId = record.id;
    expect(record.kind).toBe("matrix");
    await page
      .getByRole("link", { name: "Open notebook", exact: true })
      .click();
    await page.getByLabel("Experiment type").selectOption("matrix");
    await page.getByRole("link", { name: /Saved embedding geometry/ }).click();
    await expect(
      page.getByRole("heading", { name: metrics.name, exact: true }),
    ).toBeVisible();
    await page
      .getByLabel("Observations")
      .fill("Revisit the complete spectrum before interpreting rank.");
    await page.getByRole("button", { name: "Save notes", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Changes saved", exact: true }),
    ).toBeVisible();
    expect((await request.delete("/api/runtime")).status()).toBe(204);
    unloaded = true;
    await page.reload();
    await expect(page.getByLabel("Observations")).toHaveValue(
      "Revisit the complete spectrum before interpreting rank.",
    );
    await expect(
      page.getByText("No model loaded", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Matrix properties", exact: true }),
    ).toBeVisible();
    expect(await exported(page)).toEqual(metrics);
    const stored = await (
      await request.get("/api/experiments/" + recordId)
    ).json();
    expect(stored.result).toEqual(metrics);
    expect(stored.trace).toBeUndefined();
    const retry = await request.post("/api/experiments/from-matrix", {
      data: {
        analysis_id: metrics.analysis_id,
        name: "Old name",
        expected_execution_id: metrics.execution_id,
      },
    });
    expect(retry.status()).toBe(201);
    expect((await retry.json()).id).toBe(recordId);
    expect(
      (await (await request.get("/api/experiments/" + recordId)).json()).notes,
    ).toBe(stored.notes);
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await capturePage(page, {
      path: "../artifacts/screenshots/saved-matrix.png",
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await capturePage(page, {
      path: "../artifacts/screenshots/saved-matrix-mobile.png",
      fullPage: true,
    });
  } finally {
    if (recordId) await request.delete("/api/experiments/" + recordId);
    if (unloaded) await restore(request, before);
  }
});

test("an expired matrix save keeps its source analysis available for export", async ({
  page,
  request,
}) => {
  const before: RuntimeState = await (await request.get("/api/runtime")).json();
  const metrics = await inspect(page, "h.0.attn.c_proj.weight");
  const recordsBefore = await (await request.get("/api/experiments")).json();
  expect((await request.delete("/api/runtime")).status()).toBe(204);
  try {
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    await expect(page.getByRole("main").getByRole("alert")).toContainText(
      "Reopen the matrix",
    );
    expect(await exported(page)).toEqual(metrics);
    expect(await (await request.get("/api/experiments")).json()).toEqual(
      recordsBefore,
    );
  } finally {
    await restore(request, before);
  }
});

test("a lost matrix save response returns the original record and preserves edits on retry", async ({
  page,
  request,
}) => {
  const metrics = await inspect(page, "h.0.attn.c_proj.weight");
  let recordId: string | undefined;
  await page.route("**/api/experiments/from-matrix", async (route) => {
    const response = await route.fetch();
    const record = await response.json();
    recordId = record.id;
    await request.patch("/api/experiments/" + record.id, {
      data: {
        name: "Edited matrix before retry",
        notes: "Keep the original measurements.",
      },
    });
    await route.abort("failed");
  });
  try {
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    await expect(page.getByRole("main").getByRole("alert")).toBeVisible();
    await page.unroute("**/api/experiments/from-matrix");
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    await expect(
      page.getByRole("button", { name: "Saved to experiments", exact: true }),
    ).toBeDisabled();
    await expect(page.getByLabel("Experiment name")).toHaveValue(
      "Edited matrix before retry",
    );
    const records = await (await request.get("/api/experiments")).json();
    expect(
      records.filter(
        (record: { source_analysis_id?: string }) =>
          record.source_analysis_id === metrics.analysis_id,
      ),
    ).toHaveLength(1);
    const stored = await (
      await request.get("/api/experiments/" + recordId)
    ).json();
    expect(stored.result).toEqual(metrics);
    expect(stored.notes).toBe("Keep the original measurements.");
  } finally {
    await page.unrouteAll({ behavior: "wait" });
    if (recordId) await request.delete("/api/experiments/" + recordId);
  }
});

test("selecting another matrix starts a separate save form tied to its own measurements", async ({
  page,
  request,
}) => {
  const ids: string[] = [];
  try {
    const first = await inspect(page, "h.0.attn.c_proj.weight");
    const saving = page.waitForResponse((response) =>
      response.url().endsWith("/api/experiments/from-matrix"),
    );
    await page.getByLabel("Experiment name").fill("First matrix snapshot");
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    ids.push((await (await saving).json()).id);
    await expect(
      page.getByRole("button", { name: "Saved to experiments", exact: true }),
    ).toBeVisible();
    await page.getByLabel("Filter matrices").fill("h.1.attn.c_proj.weight");
    const analyzing = page.waitForResponse(
      (response) => response.url().includes("/api/matrix?") && response.ok(),
    );
    await page.locator(".weight-item").first().click();
    const second = await (await analyzing).json();
    expect(second.analysis_id).not.toBe(first.analysis_id);
    await expect(page.getByLabel("Experiment name")).toHaveValue(second.name);
    await expect(
      page.getByRole("button", { name: "Save experiment", exact: true }),
    ).toBeEnabled();
    const saved = page.waitForResponse((response) =>
      response.url().endsWith("/api/experiments/from-matrix"),
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    const record = await (await saved).json();
    ids.push(record.id);
    expect(
      (await (await request.get("/api/experiments/" + record.id)).json())
        .result,
    ).toEqual(second);
  } finally {
    for (const id of ids) await request.delete("/api/experiments/" + id);
  }
});
