import AxeBuilder from "@axe-core/playwright";
import { expect, test as base, type APIRequestContext } from "@playwright/test";
import { mkdirSync } from "node:fs";
import type { RuntimeState } from "../src/lib/types";
import type { PromptSetResult } from "../src/lib/prompt-sets";

const pairs = ["the", "The"].flatMap((article) => [
  {
    label: "Literal {raw}",
    clean_prompt: `${article} capital of France`,
    corrupted_prompt: `${article} capital of Poland`,
    target: "France",
    alternative: "Poland",
  },
  {
    label: "Counterexample {raw}",
    clean_prompt: `${article} capital of Poland`,
    corrupted_prompt: `${article} capital of France`,
    target: "Poland",
    alternative: "France",
  },
]);
async function load(
  request: APIRequestContext,
  settings: Record<string, unknown>,
) {
  const response = await request.post("/api/models/load", { data: settings });
  expect(response.status()).toBe(202);
  const job = await response.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + job.id)).json()).status,
    )
    .toBe("ready");
}
const test = base.extend<{ modelSession: void }>({
  modelSession: [
    async ({ request }, use) => {
      const before: RuntimeState = await (
        await request.get("/api/runtime")
      ).json();
      await load(request, {
        model_id: "hf-internal-testing/tiny-random-gpt2",
        device: "cpu",
        precision: "float32",
        context_length: 128,
      });
      try {
        await use();
      } finally {
        if (before.loaded && before.metadata)
          await load(request, {
            model_id: before.metadata.id,
            revision: before.metadata.revision,
            device: before.device,
            precision: before.precision || "auto",
            context_length:
              before.context_length ||
              before.metadata.execution_context_length ||
              128,
          });
        else await request.delete("/api/runtime");
      }
    },
    { auto: true },
  ],
});

test("measure a prompt set, inspect disagreement and retain reproducible evidence offline", async ({
  page,
  request,
}) => {
  let savedId = "";
  try {
    await page.goto("/causal");
    await page.getByRole("button", { name: "Prompt set", exact: true }).click();
    const panel = page.locator("section.panel").filter({
      has: page.getByRole("heading", {
        name: "Prompt-set patching",
        exact: true,
      }),
    });
    await panel
      .getByLabel("Import prompt pairs", { exact: true })
      .setInputFiles({
        name: "pairs.json",
        mimeType: "application/json",
        buffer: Buffer.from(JSON.stringify(pairs)),
      });
    await panel
      .getByText("Layers and reproducibility", { exact: true })
      .click();
    const runtime: RuntimeState = await (
      await request.get("/api/runtime")
    ).json();
    const last = runtime.metadata!.num_layers - 1;
    await panel.getByLabel("Layer indices", { exact: true }).fill(`0, ${last}`);
    await panel
      .getByLabel("Positions to restore", { exact: true })
      .selectOption("all");
    const run = panel.getByRole("button", {
      name: "Run prompt set",
      exact: true,
    });
    await expect(run).toBeDisabled();
    const checking = page.waitForResponse((r) =>
      r.url().endsWith("/api/causal/sets/plan"),
    );
    await panel
      .getByRole("button", { name: "Check prompt set", exact: true })
      .click();
    const plan = await (await checking).json();
    expect(plan.compatible).toBe(true);
    expect(plan.forward_passes).toBe(16);
    await expect(run).toBeEnabled();
    await run.click();
    const results = page.locator(".set-results");
    await expect(results).toContainText("4 of 4 pairs completed");
    await results
      .getByLabel("Inspect layer", { exact: true })
      .selectOption(String(last));
    await results
      .getByText("Inspect captured pair", { exact: true })
      .first()
      .click();
    await expect(results.locator(".set-pair-evidence").first()).toContainText(
      "the capital of",
    );
    const downloading = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export prompt set", exact: true })
      .click();
    const stream = await (await downloading).createReadStream();
    const buffers = [];
    for await (const chunk of stream!) buffers.push(chunk);
    const captured: PromptSetResult = JSON.parse(
      Buffer.concat(buffers).toString(),
    );
    expect(captured.metric).toBe("logit_difference");
    expect(captured.completed_pairs).toBe(4);
    expect(captured.selected_layers).toEqual([0, last]);
    expect(captured.statistics.bootstrap_samples).toBe(2000);
    expect(captured.layers.every((layer) => layer.mean_interval !== null)).toBe(
      true,
    );
    for (const pair of captured.pairs)
      expect(pair.result.layers[1].metric_delta).toBeCloseTo(
        pair.baseline_gap,
        5,
      );
    await page
      .getByLabel("Experiment name", { exact: true })
      .fill("Prompt set {literal}");
    const saving = page.waitForResponse((r) =>
      r.url().endsWith("/api/experiments/from-job"),
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    savedId = (await (await saving).json()).id;
    await request.delete("/api/runtime");
    await page
      .getByRole("link", { name: "View saved experiment", exact: true })
      .click();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(
      "Prompt set {literal}",
    );
    await expect(page.locator(".set-results")).toContainText("Literal {raw}");
    const stored = await (
      await request.get("/api/experiments/" + savedId)
    ).json();
    expect(stored.kind).toBe("causal_batch");
    expect(stored.result).toEqual(captured);
    await page.getByRole("link", { name: "Experiments", exact: true }).click();
    await page
      .getByLabel("Experiment type", { exact: true })
      .selectOption("causal_batch");
    await expect(
      page.getByRole("heading", { name: "Prompt set {literal}", exact: true }),
    ).toBeVisible();
  } finally {
    if (savedId) await request.delete("/api/experiments/" + savedId);
  }
});

test("invalid imports preserve drafts and invalid pairs cannot start a forward job", async ({
  page,
}) => {
  await page.goto("/causal");
  await page.getByRole("button", { name: "Prompt set", exact: true }).click();
  const panel = page.locator("section.panel").filter({
    has: page.getByRole("heading", {
      name: "Prompt-set patching",
      exact: true,
    }),
  });
  const importer = panel.getByLabel("Import prompt pairs", { exact: true });
  await importer.setInputFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify([{ ...pairs[0], typo: "must not disappear" }]),
    ),
  });
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "unknown fields",
  );
  await expect(panel.locator(".set-pair")).toHaveCount(4);
  const invalid = [
    { ...pairs[0], target: "France Poland" },
    { ...pairs[1], corrupted_prompt: "the" },
  ];
  await importer.setInputFiles({
    name: "invalid-pairs.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(invalid)),
  });
  await panel
    .getByRole("button", { name: "Check prompt set", exact: true })
    .click();
  await expect(panel.locator(".set-pair.invalid")).toHaveCount(2);
  await expect(
    panel.locator(".set-pair").nth(0).getByRole("alert"),
  ).toContainText("exactly one token");
  await expect(
    panel.locator(".set-pair").nth(1).getByRole("alert"),
  ).toContainText("same nonzero length");
  let requests = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/api/causal/sets/run")) requests++;
  });
  const run = panel.getByRole("button", {
    name: "Run prompt set",
    exact: true,
  });
  await expect(run).toBeDisabled();
  await run.focus();
  await expect(run).toHaveAccessibleDescription(
    "Resolve the validation and resource blockers before running.",
  );
  await run.press("Enter");
  await run.click({ force: true });
  expect(requests).toBe(0);
  await panel
    .getByRole("button", { name: "Add prompt pair", exact: true })
    .click();
  const added = panel.locator(".set-pair").last();
  await expect(added).toHaveAttribute("open", "");
  await added.getByLabel("Clean prompt", { exact: true }).fill("the capital");
  await expect(added).toHaveAttribute("open", "");
  await page.getByRole("button", { name: "Single pair", exact: true }).click();
  await page.getByRole("button", { name: "Prompt set", exact: true }).click();
  await expect(panel.locator(".set-pair")).toHaveCount(3);
  await page.reload();
  await page.getByRole("button", { name: "Prompt set", exact: true }).click();
  await expect(panel.locator(".set-pair")).toHaveCount(3);
  await panel.locator(".set-pair summary").last().click();
  await expect(
    panel
      .locator(".set-pair")
      .last()
      .getByLabel("Clean prompt", { exact: true }),
  ).toHaveValue("the capital");
  await expect(
    panel
      .locator(".set-pair")
      .last()
      .getByLabel("Corrupted prompt", { exact: true }),
  ).toHaveValue("");
});

test("prompt-set results stay accessible and readable in all locales and themes", async ({
  browser,
  request,
  baseURL,
}) => {
  test.setTimeout(180000);
  const runtime: RuntimeState = await (
    await request.get("/api/runtime")
  ).json();
  const cohortPairs = [
    ...pairs.slice(0, 3),
    ...Array.from({ length: 9 }, (_, i) => ({
      ...pairs[0],
      label: `Reverse ${i}`,
      clean_prompt: pairs[0].corrupted_prompt,
      corrupted_prompt: pairs[0].clean_prompt,
    })),
  ];
  const response = await request.post("/api/causal/sets/run", {
    data: {
      expected_execution_id: runtime.metadata!.execution_id,
      pairs: cohortPairs,
      layers: [0, runtime.metadata!.num_layers - 1],
      position_mode: "last",
      metric: "logit_difference",
      seed: 7,
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
  const measured = (await (await request.get("/api/jobs/" + job.id)).json())
    .result;
  expect(measured.completed_pairs).toBe(12);
  expect(measured.recovery_pairs).toBe(3);
  mkdirSync("../artifacts/screenshots", { recursive: true });
  for (const locale of ["en", "fa", "ar", "ja", "fr"])
    for (const theme of ["light", "dark"]) {
      const context = await browser.newContext({
        baseURL,
        viewport: { width: 390, height: 844 },
        hasTouch: true,
      });
      try {
        await context.addCookies([
          { name: "lantern_locale", value: locale, url: baseURL! },
          { name: "lantern_theme", value: theme, url: baseURL! },
        ]);
        const page = await context.newPage();
        await page.addInitScript(
          (id) => sessionStorage.setItem("lantern-prompt-set-job", id),
          job.id,
        );
        await page.goto("/causal");
        await page.locator(".scope-switch button").nth(1).click();
        await expect(page.locator(".set-results")).toBeVisible();
        await expect(page.locator(".set-results")).toContainText(
          "Literal {raw}",
        );
        if (locale === "en" && theme === "light") {
          const caution = page.getByText(
            "Small prompt sets give unstable uncertainty estimates.",
            { exact: true },
          );
          await expect(caution).toHaveCount(0);
          await page
            .getByLabel("Result measure", { exact: true })
            .selectOption("recovery");
          await expect(caution).toBeVisible();
          await page
            .getByLabel("Result measure", { exact: true })
            .selectOption("effect");
          await expect(caution).toHaveCount(0);
        }
        await page.locator(".set-pair-evidence summary").first().click();
        await page.setViewportSize({ width: 320, height: 740 });
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
        await page.setViewportSize({ width: 390, height: 844 });
        await page.locator(".set-results").evaluate((element) =>
          window.scrollTo({
            top: element.getBoundingClientRect().top + scrollY - 100,
            behavior: "instant",
          }),
        );
        await page.evaluate(() => document.fonts.ready);
        await page.mouse.move(4, 4);
        await page.screenshot({
          path: `../artifacts/screenshots/prompt-set-${locale}-${theme}.png`,
        });
      } finally {
        await context.close();
      }
    }
});
