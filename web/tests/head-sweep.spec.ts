import { expect, test, type APIRequestContext } from "@playwright/test";
import type { HeadSweepResult, Job } from "../src/lib/types";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

test("head sweep compares independent heads, filters results and restores an exported session", async ({
  page,
}) => {
  await page.goto("/causal");
  await page.getByRole("link", { name: "Head ablation", exact: true }).click();
  await expect(page).toHaveTitle("Head ablation · Circuit Lantern");
  await expect(page.getByRole("link", { name: "Causal tracing" })).toHaveClass(
    /active/,
  );
  await page.getByLabel("Layers to test").selectOption("all");
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  const table = page.locator(".head-sweep-table table");
  await expect(table.locator("tbody tr")).toHaveCount(20);
  await page.getByLabel("Head results order").selectOption("position");
  await expect(
    table.locator("tbody tr").first().getByRole("cell").nth(0),
  ).toHaveText("0");
  await expect(
    table.locator("tbody tr").first().getByRole("cell").nth(1),
  ).toHaveText("0");
  await page.getByLabel("Head results layer").selectOption("2");
  await expect(table.locator("tbody tr")).toHaveCount(4);
  await page.getByText("Captured prompt and method", { exact: true }).click();
  await capturePage(page, {
    path: "../artifacts/screenshots/head-sweep-desktop.png",
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export head sweep" }).click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const result = JSON.parse(Buffer.concat(chunks).toString());
  expect(result.complete).toBe(true);
  expect(result.error).toBeNull();
  expect(result.heads).toHaveLength(20);
  expect(result.scope).toBe("all_prompt_tokens");
  expect(result.execution_id).toMatch(/^[a-f0-9]{32}$/);
  expect(
    result.heads.every(
      (head: { kl_divergence: number }) => head.kl_divergence >= 0,
    ),
  ).toBe(true);
  const cell = table.locator("tbody tr").first();
  const head = result.heads.find(
    (head: { layer: number; head: number }) =>
      head.layer === 2 && head.head === 0,
  );
  expect(Number(await cell.getByRole("cell").nth(2).textContent())).toBeCloseTo(
    head.delta_log_probability,
    5,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/head-sweep-mobile.png",
  });
  await page
    .getByRole("link", { name: "Residual patching", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Restore a signal. Measure the effect.",
    }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Head ablation", exact: true }).click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel("Layers to test")).toHaveValue("all");
});

test("head controls explain unsupported layouts and disable oversized selections", async ({
  page,
}) => {
  await page.route("**/api/runtime", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      json: { ...(await response.json()), max_head_sweep_heads: 5 },
    });
  });
  await page.goto("/causal/heads");
  await expect(
    page.getByRole("option", { name: /All layers/ }),
  ).toHaveJSProperty("disabled", true);
  await expect(
    page.getByRole("button", { name: "Run head sweep", exact: true }),
  ).toBeEnabled();
  await page.unroute("**/api/runtime");
  await page.route("**/api/runtime", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      json: { ...(await response.json()), attention_head_targets: [] },
    });
  });
  await page.getByRole("button", { name: "Refresh runtime" }).click();
  await expect(
    page.getByText(/Head sweeps are unavailable for this architecture/),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run head sweep", exact: true }),
  ).toHaveCount(0);
});

async function seedSweep(request: APIRequestContext) {
  const queued: Job = await (
    await request.post("/api/head-sweep", {
      data: { prompt: "the capital", layers: [0] },
    })
  ).json();
  let completed: (Job & { result?: HeadSweepResult }) | undefined;
  await expect
    .poll(async () => {
      completed = await (await request.get(`/api/jobs/${queued.id}`)).json();
      return completed?.status;
    })
    .toBe("ready");
  return { queued, captured: completed!.result! };
}

for (const outcome of ["cancelled", "failed"] as const) {
  test(`a ${outcome} head sweep retains completed rows and exports a partial result`, async ({
    page,
    request,
  }) => {
    const { queued, captured } = await seedSweep(request);
    let stopped = outcome === "failed";
    const result = {
      ...captured,
      complete: false,
      cancelled: outcome === "cancelled",
      error: outcome === "failed" ? "The next head forward failed" : null,
      failed_head: outcome === "failed" ? { layer: 0, head: 1 } : null,
      heads: captured.heads.slice(0, 1),
    };
    await page.route("**/api/head-sweep", (route) =>
      route.fulfill({
        status: 202,
        json: {
          ...queued,
          id: "controlled-head-job",
          status: "running",
          progress: 25,
        },
      }),
    );
    await page.route("**/api/jobs/controlled-head-job", (route) =>
      route.fulfill({
        json: {
          ...queued,
          id: "controlled-head-job",
          status: stopped ? outcome : "running",
          progress: 25,
          message: stopped
            ? result.error ||
              "Head sweep cancelled. Completed heads are retained."
            : "Testing heads",
          ...(stopped ? { result } : {}),
        },
      }),
    );
    await page.route("**/api/jobs/controlled-head-job/cancel", (route) => {
      stopped = true;
      return route.fulfill({
        json: { status: "running", cancel_requested: true },
      });
    });
    await page.goto("/causal/heads");
    await page
      .getByRole("button", { name: "Run head sweep", exact: true })
      .click();
    if (outcome === "cancelled")
      await page.getByRole("button", { name: "Cancel sweep" }).click();
    await expect(
      page.getByText("Partial head sweep", { exact: true }),
    ).toBeVisible();
    await expect(page.locator(".head-sweep-table tbody tr")).toHaveCount(1);
    if (outcome === "failed") {
      await expect(page.getByRole("alert").first()).toContainText(
        "The next head forward failed",
      );
      await expect(
        page.getByText(/The sweep stopped at layer 0, head 1/),
      ).toBeVisible();
    }
    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export head sweep" }).click();
    const stream = await (await download).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    const exported = JSON.parse(Buffer.concat(chunks).toString());
    expect(exported.heads).toHaveLength(1);
    expect(exported.complete).toBe(false);
    expect(exported.cancelled).toBe(outcome === "cancelled");
    expect(exported.error).toBe(result.error);
    await capturePage(page, {
      path: `../artifacts/screenshots/head-sweep-${outcome === "cancelled" ? "partial" : "failed"}.png`,
    });
  });
}

test("head sweeps reject stale sessions and refresh without losing the prompt draft", async ({
  page,
  request,
}) => {
  const runtime = await (await request.get("/api/runtime")).json();
  await page.goto("/causal/heads");
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Head sweep prompt").fill("the capital of Poland");
  const replacement = await (
    await request.post("/api/models/load", {
      data: {
        model_id: runtime.metadata.id,
        revision: runtime.metadata.revision,
        device: "cpu",
        precision: "float32",
        context_length: 64,
      },
    })
  ).json();
  await expect
    .poll(
      async () =>
        (await (await request.get(`/api/jobs/${replacement.id}`)).json())
          .status,
    )
    .toBe("ready");
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "active model changed",
  );
  await page.getByRole("button", { name: "Refresh runtime" }).click();
  await expect(page.getByLabel("Head sweep prompt")).toHaveValue(
    "the capital of Poland",
  );
  await expect(
    page.getByText(/These results belong to a previous model session/),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page.getByText(/These results belong to a previous model session/),
  ).toHaveCount(0);
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
});
