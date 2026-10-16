import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { CausalResult, Experiment, RuntimeState } from "../src/lib/types";
import { capturePage } from "./helpers";

type PatchExperiment = Experiment & { kind: "causal"; result: CausalResult };
let reference: PatchExperiment;
let selected: PatchExperiment;
let equalReference: PatchExperiment;
let equalSelected: PatchExperiment;
let unrelated: Experiment;
let before: RuntimeState;
const created: string[] = [];

async function load(request: APIRequestContext) {
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

async function capture(
  request: APIRequestContext,
  name: string,
  position: number | null,
  identical = false,
) {
  const starting = await request.post("/api/causal", {
    data: {
      clean_prompt: "the capital of France is",
      corrupted_prompt: identical
        ? "the capital of France is"
        : "the capital of Poland is",
      target: "is",
      position,
    },
  });
  expect(starting.status()).toBe(202);
  const job = await starting.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + job.id)).json()).status,
    )
    .toBe("ready");
  const saving = await request.post("/api/experiments/from-job", {
    data: { job_id: job.id, name },
  });
  expect(saving.status()).toBe(201);
  const summary = await saving.json();
  created.push(summary.id);
  return (await (
    await request.get("/api/experiments/" + summary.id)
  ).json()) as PatchExperiment;
}

test.beforeAll(async ({ request }) => {
  before = await (await request.get("/api/runtime")).json();
  reference = await capture(request, "All position patch reference", null);
  await load(request);
  selected = await capture(request, "Last position patch selected", -1);
  equalReference = await capture(request, "Equal prompt reference", null, true);
  equalSelected = await capture(request, "Equal prompt selected", -1, true);
  const trace = await (
    await request.post("/api/trace", {
      data: { prompt: "the capital of France", max_new_tokens: 0 },
    })
  ).json();
  const saving = await request.post("/api/experiments", {
    data: { name: "Unrelated activation evidence", trace },
  });
  expect(saving.status()).toBe(201);
  unrelated = await saving.json();
  created.push(unrelated.id);
  expect((await request.delete("/api/runtime")).status()).toBe(204);
});

test.afterAll(async ({ request }) => {
  for (const id of created) await request.delete("/api/experiments/" + id);
  if (before) await load(request);
});

async function exported(page: Page) {
  const pending = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export comparison", exact: true })
    .click();
  const stream = await (await pending).createReadStream(),
    chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString());
}

async function open(
  page: Page,
  selectedId = selected.id,
  referenceId = reference.id,
) {
  await page.goto(
    "/experiments?experiment=" + selectedId + "&compare=" + referenceId,
  );
}

test("saved residual patches compare positions across model sessions after actual unload", async ({
  page,
  request,
}) => {
  expect((await (await request.get("/api/runtime")).json()).loaded).toBe(false);
  expect(reference.result.execution_id).not.toBe(selected.result.execution_id);
  expect(reference.result.token_ids).toEqual(selected.result.token_ids);
  expect(reference.result.clean).toEqual(selected.result.clean);
  expect(reference.result.corrupted).toEqual(selected.result.corrupted);
  let inference = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      /\/api\/(trace|causal|head-sweep|models\/load)$/.test(request.url())
    )
      inference++;
  });
  await page.goto("/experiments?experiment=" + selected.id);
  const picker = page.getByRole("combobox", {
    name: "Reference patch sweep",
    exact: true,
  });
  await expect(
    picker.locator('option[value="' + unrelated.id + '"]'),
  ).toHaveCount(0);
  await picker.selectOption(reference.id);
  await expect(
    page.getByRole("status").filter({ hasText: "baseline scores match" }),
  ).toContainText("different model sessions");
  await expect(page.locator(".matrix-comparison-source").first()).toContainText(
    "Position 4",
  );
  await expect(page.locator(".matrix-comparison-source").last()).toContainText(
    "All token positions",
  );
  await page.getByText("Inspect per block changes", { exact: true }).click();
  await expect(page.locator(".causal-comparison-table tbody tr")).toHaveCount(
    selected.result.total_layers,
  );
  const value = await exported(page);
  expect(value.format).toBe("circuit-lantern-causal-comparison-v1");
  expect(value.selected).toEqual(selected);
  expect(value.reference).toEqual(reference);
  expect(value.layers).toHaveLength(selected.result.total_layers);
  for (const [index, row] of value.layers.entries()) {
    expect(row.layer).toBe(index);
    expect(row.selected_delta).toBe(
      selected.result.layers[index].delta_log_probability,
    );
    expect(row.reference_delta).toBe(
      reference.result.layers[index].delta_log_probability,
    );
    expect(row.delta_difference).toBe(row.selected_delta - row.reference_delta);
    expect(row.recovery_difference).toBe(
      row.selected_recovery === null || row.reference_recovery === null
        ? null
        : row.selected_recovery - row.reference_recovery,
    );
  }
  expect(
    value.layers.some(
      (row: { delta_difference: number }) => row.delta_difference !== 0,
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
    path: "../artifacts/screenshots/saved-causal-comparison.png",
  });
  const bookmark = page.url();
  await page.reload();
  await expect(picker).toHaveValue(reference.id);
  await page.getByText("Inspect per block changes", { exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
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
    path: "../artifacts/screenshots/saved-causal-comparison-mobile.png",
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

test("undefined recovery remains unavailable while actual zero effects remain zero", async ({
  page,
}) => {
  await open(page, equalSelected.id, equalReference.id);
  await expect(
    page.getByText("Normalized recovery is undefined for this prompt pair.", {
      exact: false,
    }),
  ).toBeVisible();
  const value = await exported(page);
  expect(value.layers).toHaveLength(equalSelected.result.total_layers);
  for (const row of value.layers) {
    expect(row.selected_delta).toBe(0);
    expect(row.reference_delta).toBe(0);
    expect(row.delta_difference).toBe(0);
    expect(row.selected_recovery).toBeNull();
    expect(row.reference_recovery).toBeNull();
    expect(row.recovery_difference).toBeNull();
  }
  await page.getByText("Inspect per block changes", { exact: true }).click();
  await expect(
    page
      .locator(".causal-comparison-table tbody tr")
      .first()
      .getByRole("cell")
      .nth(0),
  ).toHaveText("0");
  await expect(
    page
      .locator(".causal-comparison-table tbody tr")
      .first()
      .getByRole("cell")
      .nth(3),
  ).toHaveText("N/A");
});

for (const count of [0, 2]) {
  test(
    "a partial patch comparison preserves missing blocks with " +
      count +
      " selected measurements",
    async ({ page }) => {
      const partial = structuredClone(selected),
        other = structuredClone(reference);
      Object.assign(partial.result, {
        complete: false,
        cancelled: false,
        error: "A later forward failed",
        failed_layer: count,
        layers: partial.result.layers.slice(0, count),
      });
      Object.assign(other.result, {
        complete: false,
        cancelled: true,
        layers: other.result.layers.slice(0, 1),
      });
      await page.route("**/api/experiments/" + selected.id, (route) =>
        route.fulfill({ json: partial }),
      );
      await page.route("**/api/experiments/" + reference.id, (route) =>
        route.fulfill({ json: other }),
      );
      await open(page);
      await expect(
        page.locator(".matrix-comparison-source").first(),
      ).toContainText("Partial");
      await expect(
        page.locator(".matrix-comparison-source").last(),
      ).toContainText("Partial");
      await expect(
        page.getByText(
          Math.min(count, 1) +
            " of 5 blocks have measurements in both captures.",
          { exact: false },
        ),
      ).toBeVisible();
      const value = await exported(page);
      expect(value.selected).toEqual(partial);
      expect(value.reference).toEqual(other);
      expect(value.layers[count]).toMatchObject({
        selected_delta: null,
        selected_recovery: null,
        delta_difference: null,
        recovery_difference: null,
      });
      expect(value.layers[1].reference_delta).toBeNull();
      expect(value.layers[4]).toMatchObject({
        selected_delta: null,
        reference_delta: null,
        delta_difference: null,
      });
      await page
        .getByText("Inspect per block changes", { exact: true })
        .click();
      await expect(
        page
          .locator(".causal-comparison-table tbody tr")
          .last()
          .getByRole("cell"),
      ).toHaveText(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]);
      if (count === 0)
        await expect(
          page.getByText("No patch measurements completed.", { exact: false }),
        ).toBeVisible();
      expect(
        (
          await new AxeBuilder({ page })
            .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
            .analyze()
        ).violations,
      ).toEqual([]);
    },
  );
}

test("a single measured zero effect remains visible as a point", async ({
  page,
}) => {
  const partial = structuredClone(equalSelected);
  Object.assign(partial.result, {
    complete: false,
    cancelled: true,
    layers: partial.result.layers.slice(0, 1),
  });
  await page.route("**/api/experiments/" + equalSelected.id, (route) =>
    route.fulfill({ json: partial }),
  );
  await open(page, equalSelected.id, equalReference.id);
  const panel = page.locator("section.panel").filter({
    has: page.getByRole("heading", {
      name: "Patch effect comparison",
      exact: true,
    }),
  });
  await expect(panel.locator(".recharts-line-dot")).toHaveCount(1);
  await expect(panel.locator(".recharts-line-dot")).toBeVisible();
  const value = await exported(page);
  expect(value.layers[0]).toMatchObject({
    selected_delta: 0,
    reference_delta: 0,
    delta_difference: 0,
  });
  expect(value.layers[1].selected_delta).toBeNull();
});

test("a shortened capture without a completion flag stays visibly partial", async ({
  page,
}) => {
  const partial = structuredClone(selected);
  delete partial.result.complete;
  partial.result.layers = partial.result.layers.slice(0, 2);
  await page.route("**/api/experiments/" + selected.id, (route) =>
    route.fulfill({ json: partial }),
  );
  await open(page);
  await expect(page.locator(".matrix-comparison-source").first()).toContainText(
    "Partial",
  );
  await expect(
    page.locator(".trace-meta").getByText("Partial sweep", { exact: true }),
  ).toBeVisible();
  const value = await exported(page);
  expect(value.layers[2].selected_delta).toBeNull();
});

const variants: [string, (result: CausalResult) => void, string][] = [
  [
    "revision",
    (r) => {
      r.revision = "other-revision";
    },
    "different checkpoints",
  ],
  [
    "prompt text",
    (r) => {
      r.clean_prompt = "a different prompt";
    },
    "prompts do not align",
  ],
  [
    "clean vocabulary IDs",
    (r) => {
      r.token_ids![0]++;
    },
    "prompts do not align",
  ],
  [
    "corrupted vocabulary IDs",
    (r) => {
      r.corrupted_token_ids![0]++;
    },
    "prompts do not align",
  ],
  [
    "legacy token identity",
    (r) => {
      delete r.token_ids;
    },
    "prompt vocabulary IDs",
  ],
  [
    "target",
    (r) => {
      r.target_token_id++;
    },
    "different target tokens",
  ],
  [
    "precision",
    (r) => {
      r.execution_precision = "bfloat16";
    },
    "Execution settings differ",
  ],
  [
    "legacy execution",
    (r) => {
      delete r.execution_config;
    },
    "older capture",
  ],
  [
    "library version",
    (r) => {
      r.execution_config!.torch_version = "other-version";
    },
    "Execution settings differ",
  ],
  [
    "patch method",
    (r) => {
      r.method = "a different intervention";
    },
    "conventions differ",
  ],
  [
    "block layout",
    (r) => {
      r.total_layers++;
      r.complete = false;
    },
    "conventions differ",
  ],
  [
    "baseline",
    (r) => {
      r.clean.log_probability -= 0.01;
    },
    "Baseline scores differ",
  ],
  [
    "missing score",
    (r) => {
      Object.assign(r.layers[0], { delta_log_probability: null });
    },
    "finite patch measurements",
  ],
  [
    "duplicate block",
    (r) => {
      r.layers[1].layer = r.layers[0].layer;
    },
    "finite patch measurements",
  ],
];

for (const [name, change, message] of variants) {
  test(
    "patch comparison refuses incompatible " +
      name +
      " without losing the selected sweep",
    async ({ page }) => {
      const other = structuredClone(reference);
      change(other.result);
      await page.route("**/api/experiments/" + reference.id, (route) =>
        route.fulfill({ json: other }),
      );
      await open(page);
      await expect(
        page.locator(".trace-comparison").getByRole("alert"),
      ).toContainText(message);
      await expect(
        page.getByRole("button", { name: "Export comparison", exact: true }),
      ).toHaveCount(0);
      await expect(
        page.getByRole("heading", {
          name: "Per block patching results",
          exact: true,
        }),
      ).toBeVisible();
      await expect(page.locator(".causal-results-table tbody tr")).toHaveCount(
        selected.result.layers.length,
      );
    },
  );
}

test("self links, nonpatch references and missing records can be cleared while preserving notebook filters", async ({
  page,
}) => {
  for (const [id, message] of [
    [selected.id, "different saved residual patch sweep"],
    [unrelated.id, "not a residual patch sweep"],
    ["0".repeat(32), "not found"],
  ]) {
    await page.goto(
      "/experiments?q=patch&kind=causal&experiment=" +
        selected.id +
        "&compare=" +
        id,
    );
    await expect(
      page.locator(".trace-comparison").getByRole("alert"),
    ).toContainText(message);
    await page
      .getByRole("combobox", { name: "Reference patch sweep", exact: true })
      .selectOption("");
    await expect(
      page.locator(".trace-comparison").getByRole("alert"),
    ).toHaveCount(0);
    await expect(
      page.getByRole("heading", {
        name: "Per block patching results",
        exact: true,
      }),
    ).toBeVisible();
  }
  await page
    .getByRole("link", { name: "All experiments", exact: true })
    .click();
  await expect(
    page.getByRole("searchbox", { name: "Search notebook", exact: true }),
  ).toHaveValue("patch");
  await expect(page).not.toHaveURL(/compare=/);
});
