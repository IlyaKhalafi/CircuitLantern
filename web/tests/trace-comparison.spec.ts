import { expect, test, type APIRequestContext } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Experiment, RuntimeState, Trace } from "../src/lib/types";
import { capturePage } from "./helpers";

type TraceExperiment = Experiment & { trace: Trace };
let baseline: TraceExperiment;
let selected: TraceExperiment;
let otherPrompt: TraceExperiment;
let headSweep: Experiment;
const created: string[] = [];

async function capture(
  request: APIRequestContext,
  name: string,
  extra: Record<string, unknown> = {},
) {
  const response = await request.post("/api/trace", {
    data: {
      prompt: "the capital of France",
      max_new_tokens: 0,
      capture_attention: true,
      ...extra,
    },
  });
  expect(response.ok()).toBe(true);
  const trace = await response.json();
  const saved = await request.post("/api/experiments", {
    data: { name, trace },
  });
  expect(saved.status()).toBe(201);
  const summary = await saved.json();
  created.push(summary.id);
  return (await (
    await request.get("/api/experiments/" + summary.id)
  ).json()) as TraceExperiment;
}

test.beforeAll(async ({ request }) => {
  baseline = await capture(request, "Comparison reference baseline");
  const runtime: RuntimeState = await (
    await request.get("/api/runtime")
  ).json();
  const response = await request.post("/api/models/load", {
    data: {
      model_id: runtime.metadata!.id,
      revision: runtime.metadata!.revision,
      context_length: runtime.context_length,
      precision: runtime.precision,
      device: runtime.device,
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
  selected = await capture(request, "Comparison selected head intervention", {
    max_new_tokens: 1,
    ablate_head: { layer: 0, head: 1 },
  });
  otherPrompt = await capture(request, "Comparison unrelated prompt", {
    prompt: "the opposite of hot",
  });
  const started = await request.post("/api/head-sweep", {
    data: { prompt: baseline.trace.prompt, layers: [0] },
  });
  expect(started.status()).toBe(202);
  const sweep = await started.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + sweep.id)).json()).status,
    )
    .toBe("ready");
  const saved = await request.post("/api/experiments/from-job", {
    data: { name: "Comparison nontrace evidence", job_id: sweep.id },
  });
  expect(saved.status()).toBe(201);
  headSweep = await saved.json();
  created.push(headSweep.id);
});

test.afterAll(async ({ request }) => {
  for (const id of created) await request.delete("/api/experiments/" + id);
});

test("saved traces compare across matching executions and export exact source evidence", async ({
  page,
  request,
}) => {
  expect(selected.trace.execution_id).not.toBe(baseline.trace.execution_id);
  expect(selected.trace.generation_performed).toBe(true);
  expect(baseline.trace.generation_performed).toBe(false);
  let inference = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      /\/api\/(trace|causal|head-sweep|models\/load)$/.test(request.url())
    )
      inference++;
  });
  await page.route("**/api/runtime", (route) =>
    route.fulfill({ json: { loaded: false, metadata: null, device: "cpu" } }),
  );
  await page.goto("/experiments?experiment=" + selected.id);
  await page
    .getByRole("combobox", { name: "Reference trace", exact: true })
    .selectOption(baseline.id);
  await expect(
    page.getByRole("status").filter({ hasText: "execution settings match" }),
  ).toContainText("different model sessions");
  await expect(
    page.getByRole("button", { name: "Export comparison", exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator(".recharts-legend-wrapper")
      .getByText("Reference trace", { exact: true }),
  ).toHaveCount(2);
  await page.getByText("Inspect per-layer changes", { exact: true }).click();
  await expect(page.locator(".trace-comparison-table tbody tr")).toHaveCount(
    selected.trace.layers.length,
  );
  await expect(page.locator(".comparison-values")).toContainText(
    "do not give the distance between residual vectors",
  );
  const downloading = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export comparison", exact: true })
    .click();
  const download = await downloading;
  const stream = await download.createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const exported = JSON.parse(Buffer.concat(chunks).toString());
  expect(exported.format).toBe("circuit-lantern-trace-comparison-v1");
  expect(exported.selected).toEqual(selected);
  expect(exported.reference).toEqual(baseline);
  expect(exported.layers).toHaveLength(selected.trace.layers.length);
  for (const [index, row] of exported.layers.entries()) {
    expect(row.layer).toBe(selected.trace.layers[index].layer);
    expect(row.norm_difference).toBe(
      selected.trace.layers[index].norm - baseline.trace.layers[index].norm,
    );
    expect(row.entropy_difference).toBe(
      selected.trace.layers[index].entropy -
        baseline.trace.layers[index].entropy,
    );
  }
  expect(
    exported.layers.some(
      (row: { norm_difference: number }) => row.norm_difference !== 0,
    ),
  ).toBe(true);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await capturePage(page, {
    path: "../artifacts/screenshots/saved-trace-comparison.png",
    fullPage: true,
  });
  const bookmark = page.url();
  await page.reload();
  await expect(
    page.getByRole("combobox", { name: "Reference trace", exact: true }),
  ).toHaveValue(baseline.id);
  await expect(
    page.getByRole("button", { name: "Export comparison", exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/saved-trace-comparison-mobile.png",
    fullPage: true,
  });
  await page
    .getByRole("combobox", { name: "Reference trace", exact: true })
    .selectOption("");
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
    await (await request.get("/api/experiments/" + baseline.id)).json(),
  ).toEqual(baseline);
});

test("unrelated prompts and nontrace links keep the selected capture readable", async ({
  page,
}) => {
  for (const [record, message] of [
    [otherPrompt, "prompts do not align"],
    [headSweep, "not an activation trace"],
    [selected, "different saved trace"],
  ] as const) {
    await page.goto(
      "/experiments?experiment=" + selected.id + "&compare=" + record.id,
    );
    await expect(
      page.locator(".trace-comparison").getByRole("alert"),
    ).toContainText(message);
    await expect(
      page.getByRole("button", { name: "Export comparison", exact: true }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("heading", { name: "Tokenized prompt", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("combobox", { name: "Reference trace", exact: true })
      .selectOption("");
    await expect(
      page.locator(".trace-comparison").getByRole("alert"),
    ).toHaveCount(0);
  }
});

const variants: [string, (trace: Trace) => void, string][] = [
  [
    "missing norm",
    (trace) => {
      Object.assign(trace.layers[0], { norm: null });
    },
    "finite residual norms",
  ],
  [
    "missing entropy",
    (trace) => {
      Object.assign(trace.layers[0], { entropy: null });
    },
    "finite residual norms",
  ],
  [
    "missing layers",
    (trace) => {
      Object.assign(trace, { layers: null });
    },
    "finite residual norms",
  ],
  [
    "revision",
    (trace) => {
      trace.revision = "other-revision";
    },
    "different checkpoints",
  ],
  [
    "token IDs",
    (trace) => {
      trace.token_ids = [...trace.token_ids, 0];
    },
    "prompts do not align",
  ],
  [
    "precision",
    (trace) => {
      trace.execution_precision = "bfloat16";
    },
    "Execution settings differ",
  ],
  [
    "library version",
    (trace) => {
      trace.execution_config!.torch_version = "other-version";
    },
    "Execution settings differ",
  ],
  [
    "attention capture",
    (trace) => {
      trace.capture_attention = false;
    },
    "Attention capture settings differ",
  ],
  [
    "normalization",
    (trace) => {
      trace.final_state_normalized = !trace.final_state_normalized;
    },
    "measurement conventions differ",
  ],
  [
    "layer positions",
    (trace) => {
      trace.layers = trace.layers.slice(1);
    },
    "layer positions do not align",
  ],
  [
    "legacy provenance",
    (trace) => {
      delete trace.execution_config;
    },
    "older capture",
  ],
];
for (const [name, change, message] of variants) {
  test("comparison refuses incompatible " + name, async ({ page }) => {
    const reference = structuredClone(baseline);
    change(reference.trace);
    await page.route("**/api/experiments/" + baseline.id, (route) =>
      route.fulfill({ json: reference }),
    );
    await page.goto(
      "/experiments?experiment=" + selected.id + "&compare=" + baseline.id,
    );
    await expect(
      page.locator(".trace-comparison").getByRole("alert"),
    ).toContainText(message);
    await expect(
      page.getByRole("button", { name: "Export comparison", exact: true }),
    ).toHaveCount(0);
    await expect(page.locator(".trace-comparison-table")).toHaveCount(0);
  });
}

test("profile object key order does not block matching recorded settings", async ({
  page,
}) => {
  const reference = structuredClone(baseline);
  reference.trace.execution_config = Object.fromEntries(
    Object.entries(reference.trace.execution_config!).reverse(),
  );
  await page.route("**/api/experiments/" + baseline.id, (route) =>
    route.fulfill({ json: reference }),
  );
  await page.goto(
    "/experiments?experiment=" + selected.id + "&compare=" + baseline.id,
  );
  await expect(
    page.getByRole("button", { name: "Export comparison", exact: true }),
  ).toBeVisible();
  await expect(
    page.locator(".trace-comparison").getByRole("alert"),
  ).toHaveCount(0);
});

test("a missing reference can be cleared without losing the selected trace or filters", async ({
  page,
}) => {
  await page.goto(
    "/experiments?q=Comparison&kind=trace&experiment=" +
      selected.id +
      "&compare=" +
      "0".repeat(32),
  );
  await expect(
    page.locator(".trace-comparison").getByRole("alert"),
  ).toContainText("not found");
  await page
    .getByRole("combobox", { name: "Reference trace", exact: true })
    .selectOption("");
  await expect(
    page.locator(".trace-comparison").getByRole("alert"),
  ).toHaveCount(0);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    selected.name,
  );
  await page
    .getByRole("link", { name: "All experiments", exact: true })
    .click();
  await expect(
    page.getByRole("searchbox", { name: "Search notebook" }),
  ).toHaveValue("Comparison");
  await expect(page).not.toHaveURL(/compare=/);
});

test("a slow or unavailable reference list preserves linked evidence and offers recovery", async ({
  page,
}) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/experiments", async (route) => {
    await gate;
    await route.fulfill({
      status: 502,
      json: { detail: "Saved trace list is temporarily unavailable." },
    });
  });
  try {
    await page.goto(
      "/experiments?experiment=" + selected.id + "&compare=" + baseline.id,
    );
    await expect(
      page.getByRole("button", { name: "Export comparison", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("status").filter({ hasText: "Finding saved traces" }),
    ).toBeVisible();
    await expect(
      page.getByText("Save another activation trace to compare", {
        exact: false,
      }),
    ).toHaveCount(0);
    release();
    await expect(
      page.locator(".trace-comparison").getByRole("alert"),
    ).toContainText("temporarily unavailable");
    await page.unroute("**/api/experiments");
    await page
      .getByRole("button", { name: "Retry saved traces", exact: true })
      .click();
    await expect(
      page
        .getByRole("combobox", { name: "Reference trace", exact: true })
        .getByRole("option", { name: new RegExp(baseline.name) }),
    ).toContainText(baseline.id.slice(0, 8));
    await expect(
      page.locator(".trace-comparison").getByRole("alert"),
    ).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Export comparison", exact: true }),
    ).toBeVisible();
  } finally {
    release();
    await page.unrouteAll({ behavior: "wait" });
  }
});
