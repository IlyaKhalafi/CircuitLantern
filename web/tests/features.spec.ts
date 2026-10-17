import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { RuntimeState } from "../src/lib/types";

const documents = [
  "the capital of France is",
  "the capital of Poland is",
  "Poland is the capital of Poland",
  "France is the capital of France",
  "the capital is France",
  "the capital is Poland",
  "France of the is capital",
  "Poland of the is capital",
].join("\n");

test("train, inspect, annotate and measure sparse feature circuits without code", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1000 });
  const screenshots = join(process.cwd(), "../artifacts/screenshots");
  mkdirSync(screenshots, { recursive: true });
  async function audit(name: string) {
    if (name.includes("reconstruction")) {
      await page.locator(".feature-evaluation").evaluate((element) =>
        window.scrollTo({
          top: element.getBoundingClientRect().top + scrollY - 24,
          left: 0,
          behavior: "instant",
        }),
      );
    } else if (name.includes("inspection")) {
      await page
        .getByRole("heading", {
          name: "Test features on a prompt",
          exact: true,
        })
        .evaluate((element) =>
          window.scrollTo({
            top: element.getBoundingClientRect().top + scrollY - 24,
            left: 0,
            behavior: "instant",
          }),
        );
    } else if (name.includes("circuit-")) {
      await page
        .getByRole("heading", { name: "Saved feature circuits", exact: true })
        .evaluate((element) =>
          window.scrollTo({
            top: element.getBoundingClientRect().top + scrollY - 24,
            left: 0,
            behavior: "instant",
          }),
        );
    } else {
      await page.evaluate(() =>
        window.scrollTo({ top: 0, left: 0, behavior: "instant" }),
      );
    }
    await page.evaluate(async () => {
      await document.fonts.ready;
      document.activeElement instanceof HTMLElement &&
        document.activeElement.blur();
    });
    const result = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      result.violations.map((row) => ({
        id: row.id,
        nodes: row.nodes.map((node) => ({
          target: node.target,
          summary: node.failureSummary,
        })),
      })),
    ).toEqual([]);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({ path: join(screenshots, name + ".png") });
  }
  await page.goto("/features");
  await expect(
    page.getByRole("heading", {
      name: "Train a sparse dictionary",
      exact: true,
    }),
  ).toBeVisible();
  await page.getByLabel("Training corpus", { exact: true }).fill(documents);
  await page
    .getByLabel("Dictionary name", { exact: true })
    .fill("Measured dictionary {raw}");
  await page
    .getByLabel("Transformer layer", { exact: true })
    .selectOption("all");
  await page.getByText("Advanced training settings", { exact: true }).click();
  await page.getByLabel("Training steps", { exact: true }).fill("16");
  await page
    .getByRole("button", { name: "Check training resources", exact: true })
    .click();
  const train = page.getByRole("button", {
    name: "Train dictionary",
    exact: true,
  });
  await expect(train).toBeEnabled();
  await audit("features-training-light");
  await page.getByRole("button", { name: "Dark theme", exact: true }).click();
  await audit("features-training-dark");
  await page.getByRole("button", { name: "Light theme", exact: true }).click();
  await train.click();
  await expect(
    page.getByRole("tab", { name: "Explore features", exact: true }),
  ).toHaveAttribute("aria-selected", "true", { timeout: 60000 });
  await expect(
    page.getByRole("heading", { name: "Feature dictionary", exact: true }),
  ).toBeVisible();
  await page.locator(".feature-row").first().click();
  await page
    .getByLabel("Feature annotation", { exact: true })
    .fill("Geography {raw}");
  await page
    .getByLabel("Research notes", { exact: true })
    .fill("Literal observation, not a semantic claim.");
  await page
    .getByRole("button", { name: "Save annotation", exact: true })
    .click();
  await expect(
    page.locator(".feature-row").filter({ hasText: "Geography {raw}" }),
  ).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 0));
  await audit("features-dictionary-light");
  await page.getByRole("button", { name: "Dark theme", exact: true }).click();
  await audit("features-dictionary-dark");
  await page.getByRole("button", { name: "Light theme", exact: true }).click();
  const downloading = page.waitForEvent("download");
  await page
    .getByRole("link", { name: "Download SAE weights", exact: true })
    .click();
  expect((await downloading).suggestedFilename()).toMatch(
    /^sae-[a-f0-9]{32}\.safetensors$/,
  );
  await page
    .getByLabel("Feature prompt", { exact: true })
    .fill("the capital of France");
  await page
    .getByRole("button", { name: "Inspect features", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Baseline predictions", exact: true }),
  ).toBeVisible();
  const inspectedResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/inspect") &&
      response.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Ablate feature", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Intervened predictions", exact: true }),
  ).toBeVisible();
  const inspected = await (await inspectedResponse).json();
  const evaluatedResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/evaluate") &&
      response.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Evaluate reconstruction", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Reconstruction faithfulness",
      exact: true,
    }),
  ).toBeVisible();
  const evaluated = await (await evaluatedResponse).json();
  await page
    .getByRole("heading", { name: "Test features on a prompt", exact: true })
    .evaluate((element) =>
      window.scrollTo(0, element.getBoundingClientRect().top + scrollY - 30),
    );
  await audit("features-inspection-light");
  await audit("features-reconstruction-light");
  await page.getByRole("button", { name: "Dark theme", exact: true }).click();
  await audit("features-inspection-dark");
  await audit("features-reconstruction-dark");
  await page.getByRole("button", { name: "Light theme", exact: true }).click();
  await page
    .getByRole("tab", { name: "Feature circuits", exact: true })
    .click();
  const choices = page.locator(".feature-dictionary-choices input");
  await expect(choices).toHaveCount(5);
  for (const input of await choices.all()) await input.check();
  await page
    .getByLabel("Circuit prompt", { exact: true })
    .fill("the capital of France");
  await page
    .getByLabel("Features per layer", { exact: true })
    .selectOption("1");
  await page
    .getByRole("button", { name: "Measure feature circuit", exact: true })
    .click();
  await expect(page.locator(".feature-effects>div")).toHaveCount(5, {
    timeout: 60000,
  });
  await page
    .getByRole("heading", { name: "Saved feature circuits", exact: true })
    .evaluate((element) =>
      window.scrollTo(0, element.getBoundingClientRect().top + scrollY - 30),
    );
  await audit("features-circuit-light");
  await page.getByRole("button", { name: "Dark theme", exact: true }).click();
  await audit("features-circuit-dark");
  await page.getByRole("button", { name: "Light theme", exact: true }).click();
  const before = (await (
    await request.get("/api/runtime")
  ).json()) as RuntimeState;
  expect((await request.delete("/api/runtime")).status()).toBe(204);
  try {
    await page
      .getByRole("tab", { name: "Explore features", exact: true })
      .click();
    await page.locator('.feature-row[aria-pressed="false"]').first().click();
    await expect(page.locator(".feature-intervention-evidence")).toContainText(
      "Feature " + inspected.intervention.feature,
    );
    let forwards = 0;
    page.on("request", (req) => {
      if (req.method() === "POST" && /\/(inspect|evaluate)$/.test(req.url()))
        forwards++;
    });
    for (const [name, expected] of [
      ["Export feature inspection", inspected],
      ["Export reconstruction", evaluated],
    ] as const) {
      const pending = page.waitForEvent("download");
      await page.getByRole("button", { name, exact: true }).click();
      const download = await pending;
      const stream = await download.createReadStream();
      const chunks = [];
      for await (const chunk of stream!) chunks.push(chunk);
      const exported = JSON.parse(Buffer.concat(chunks).toString());
      expect(exported).toEqual(expected);
      expect(exported.dictionary_source.weights_sha256).toMatch(
        /^[a-f0-9]{64}$/,
      );
      expect(exported.dictionary_source.manifest_sha256).toMatch(
        /^[a-f0-9]{64}$/,
      );
      expect(exported.dictionary_source.training_execution.execution_id).toBe(
        before.metadata!.execution_id,
      );
      expect(exported.captured_at).toMatch(/T.*\+00:00$/);
      expect(download.suggestedFilename()).toMatch(
        /^feature-(inspection|reconstruction)-[a-f0-9]{32}\.json$/,
      );
    }
    expect(forwards).toBe(0);
    expect((await (await request.get("/api/runtime")).json()).loaded).toBe(
      false,
    );
  } finally {
    if (before.loaded) {
      const loading = await request.post("/api/models/load", {
        data: {
          model_id: before.metadata!.id,
          revision: before.metadata!.revision,
          device: before.device,
          precision: before.precision,
          context_length: before.context_length,
        },
      });
      expect(loading.status()).toBe(202);
      const job = await loading.json();
      await expect
        .poll(
          async () =>
            (await (await request.get("/api/jobs/" + job.id)).json()).status,
          { timeout: 60000 },
        )
        .toBe("ready");
    }
  }
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  for (const locale of ["fa", "ar", "ja", "fr", "en"]) {
    await page.locator(".language-select select").selectOption(locale);
    await expect(page.locator("html")).toHaveAttribute("lang", locale);
    await page.locator(".feature-tabs button").nth(0).click();
    await audit("features-training-" + locale + "-mobile");
    await page.locator(".feature-tabs button").nth(1).click();
    if (locale === "fa" || locale === "ar") {
      const bounds = await page
        .locator(".feature-inspection-result .metric strong")
        .first()
        .evaluate((element) => {
          const node = document
            .createTreeWalker(element, NodeFilter.SHOW_TEXT)
            .nextNode();
          const text = node?.textContent || "";
          const minus = text.indexOf("-");
          const digit = text.search(/\d/);
          if (!node || minus < 0 || digit < 0)
            throw new Error("Expected the signed fixture measurement");
          const range = document.createRange();
          range.setStart(node, minus);
          range.setEnd(node, minus + 1);
          const minusX = range.getBoundingClientRect().left;
          range.setStart(node, digit);
          range.setEnd(node, digit + 1);
          return { minusX, digitX: range.getBoundingClientRect().left };
        });
      expect(bounds.minusX).toBeLessThan(bounds.digitX);
    }
    await audit("features-reconstruction-" + locale + "-mobile");
  }
  const trainingId = await page.evaluate(() =>
    sessionStorage.getItem("lantern-feature-training-job"),
  );
  expect(trainingId).toMatch(/^[a-f0-9]{32}$/);
  let release = () => {};
  let observed = () => {};
  let completed = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const intercepted = new Promise<void>((resolve) => {
    observed = resolve;
  });
  const restored = new Promise<void>((resolve) => {
    completed = resolve;
  });
  await page.route("**/api/jobs/" + trainingId, async (route) => {
    observed();
    await gate;
    const response = await route.fetch();
    await route.fulfill({ response });
    completed();
  });
  try {
    await page.reload();
    await intercepted;
    await page
      .getByRole("tab", { name: "Feature circuits", exact: true })
      .click();
  } finally {
    release();
  }
  await restored;
  await expect(
    page.locator("#feature-tab-train .feature-job strong"),
  ).not.toBeEmpty();
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  await expect(
    page.getByRole("tab", { name: "Feature circuits", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  const saved = page.getByLabel("Open saved feature circuit", { exact: true });
  await expect(saved.locator("option")).toHaveCount(2);
  const id = await saved.locator("option").nth(1).getAttribute("value");
  await saved.selectOption(id!);
  await expect(page.locator(".feature-effects>div")).toHaveCount(5);
  await page
    .getByRole("tab", { name: "Explore features", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Delete dictionary", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Permanently delete dictionary", exact: true })
    .click();
  await expect(
    page.getByLabel("Choose a dictionary", { exact: true }).locator("option"),
  ).toHaveCount(5);
  await page
    .getByRole("tab", { name: "Feature circuits", exact: true })
    .click();
  await expect(page.locator(".feature-effects>div")).toHaveCount(5);
  expect(errors).toEqual([]);
});
