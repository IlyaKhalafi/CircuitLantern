import AxeBuilder from "@axe-core/playwright";
import { expect, type APIRequestContext, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { test, seed, load, parameters } from "./induction-fixture";
import type { Experiment } from "../src/lib/types";
import type { ProbeComparison } from "../src/lib/induction-comparison";
import english from "../src/i18n/messages/en.json";
import persian from "../src/i18n/messages/fa.json";
import arabic from "../src/i18n/messages/ar.json";
import japanese from "../src/i18n/messages/ja.json";
import french from "../src/i18n/messages/fr.json";

async function save(request: APIRequestContext, job: string, name: string) {
  const response = await request.post("/api/experiments/from-job", {
    data: { job_id: job, name },
  });
  expect(response.ok()).toBe(true);
  const summary: Experiment = await response.json();
  const retained = await request.get("/api/experiments/" + summary.id);
  expect(retained.ok()).toBe(true);
  const record: Experiment = await retained.json();
  if (record.kind !== "induction_probe" || !record.result)
    throw new Error("Expected induction evidence");
  return { ...record, result: record.result };
}

async function pair(request: APIRequestContext) {
  const reference = await seed(request);
  const a = await save(request, reference.id, "Base probe");
  await load(request, "circuit-lantern-test/finetuned");
  const response = await request.post("/api/induction/probe", {
    data: { ...parameters, reference_experiment_id: a.id },
  });
  expect(response.status()).toBe(202);
  const queued = await response.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + queued.id)).json()).status,
    )
    .toBe("ready");
  const b = await save(request, queued.id, "Finetuned probe");
  return { a, b };
}

async function screenshot(page: Page, name: string) {
  const directory = join(process.cwd(), "../artifacts/screenshots");
  mkdirSync(directory, { recursive: true });
  await page.evaluate(async () => document.fonts.ready);
  await page.screenshot({
    path: join(directory, name + ".png"),
    fullPage: false,
  });
}

async function exported(page: Page) {
  const pending = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export comparison", exact: true })
    .click();
  const stream = await (await pending).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString());
}

test("repeat saved inputs on another checkpoint and compare immutable evidence after unloading", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  const captured = await seed(request);
  const reference = await save(request, captured.id, "Base probe reference");
  let selectedId = "";
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    await page.goto("/experiments?experiment=" + reference.id);
    const comparison = page.locator(".probe-comparison");
    await expect(comparison).not.toHaveAttribute("open", "");
    await comparison.locator("summary").click();
    await page
      .getByRole("button", {
        name: "Repeat with another checkpoint",
        exact: true,
      })
      .click();
    await expect(page).toHaveURL(new RegExp("reference=" + reference.id));
    await expect(
      page.getByRole("region", { name: "Matching probe reference" }),
    ).toContainText(reference.name);
    await page.getByText("Probe settings", { exact: true }).click();
    await expect(
      page.getByRole("spinbutton", { name: "Tokens per block", exact: true }),
    ).toHaveValue("4");
    await expect(
      page.getByRole("spinbutton", { name: "Tokens per block", exact: true }),
    ).toBeDisabled();
    await page
      .getByRole("link", { name: "Choose a checkpoint", exact: true })
      .click();
    await expect(page).toHaveURL(/return=induction/);
    await page
      .getByLabel("Search Hugging Face models")
      .fill("circuit-lantern-test/finetuned");
    const checked = page.waitForRequest((request) =>
      request.url().endsWith("/api/models/compatibility"),
    );
    await page
      .getByRole("button", { name: "Check model", exact: true })
      .click();
    expect((await checked).postDataJSON()).toMatchObject({
      precision: "float32",
      context_length: 128,
      device: "cpu",
    });
    await expect(
      page.getByText("Ready to explore on this server", { exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Download & explore", exact: true })
      .click();
    await expect(page).toHaveURL(
      new RegExp("/causal/heads/induction\\?reference=" + reference.id),
    );
    await expect(
      page.getByRole("button", { name: "Run probe", exact: true }),
    ).toBeEnabled();
    const submitted = page.waitForRequest((request) =>
      request.url().endsWith("/api/induction/probe"),
    );
    await page.getByRole("button", { name: "Run probe", exact: true }).click();
    expect((await submitted).postDataJSON()).toMatchObject({
      ...parameters,
      reference_experiment_id: reference.id,
    });
    await expect(
      page.locator(".trace-meta").getByText("Probe complete", { exact: true }),
    ).toBeVisible();
    await page
      .getByLabel("Experiment name", { exact: true })
      .fill("Fine probe selection");
    const saved = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/experiments/from-job") &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    selectedId = (await (await saved).json()).id;
    await expect(
      page.getByRole("link", { name: "View saved experiment" }),
    ).toHaveAttribute(
      "href",
      "/experiments?experiment=" + selectedId + "&compare=" + reference.id,
    );
    expect((await request.delete("/api/runtime")).status()).toBe(204);
    await page.getByRole("link", { name: "View saved experiment" }).click();
    await expect(page.locator(".probe-comparison-body")).toBeVisible();
    await expect(page.locator(".probe-comparison-sources")).toContainText(
      "circuit-lantern-test/finetuned",
    );
    const evidence: ProbeComparison = await (
      await request.get(
        "/api/induction/comparison?selected=" +
          selectedId +
          "&reference=" +
          reference.id,
      )
    ).json();
    expect(evidence.shared_examples).toBe(2);
    expect(evidence.shared_pattern_heads).toBe(20);
    expect(evidence.shared_causal_heads).toBe(1);
    const selected = await (
      await request.get("/api/experiments/" + selectedId)
    ).json();
    expect(selected.result.dataset).toEqual(reference.result.dataset);
    expect(selected.result.reference_experiment_id).toBe(reference.id);
    await page.getByText("Probe setup and evidence", { exact: true }).click();
    await expect(page.locator(".induction-results")).toContainText(
      "This probe replays exact saved reference token IDs",
    );
    await expect(page.locator(".induction-results")).toContainText(
      "sampled positions use IDs designated as special by the captured checkpoint tokenizer",
    );
    await expect(
      page.getByRole("link", { name: "Open reference probe", exact: true }),
    ).toHaveAttribute("href", "/experiments?experiment=" + reference.id);
    expect(selected.result.causal_results[0].layer).toBe(
      reference.result.causal_results[0].layer,
    );
    expect(selected.result.causal_results[0].head).toBe(
      reference.result.causal_results[0].head,
    );
    await expect(page.locator(".probe-comparison-table tbody tr")).toHaveCount(
      12,
    );
    await page
      .locator(".probe-comparison-pagination")
      .getByRole("button", { name: "Next", exact: true })
      .click();
    await expect(page.locator(".probe-comparison-table tbody tr")).toHaveCount(
      8,
    );
    await page
      .getByLabel("Probe comparison layer", { exact: true })
      .selectOption("2");
    await expect(page.locator(".probe-comparison-table tbody tr")).toHaveCount(
      4,
    );
    await page
      .getByLabel("Probe comparison layer", { exact: true })
      .selectOption("all");
    await page
      .getByLabel("Probe comparison metric", { exact: true })
      .selectOption("causal");
    await expect(page.locator(".probe-comparison-table tbody tr")).toHaveCount(
      1,
    );
    await expect(page.locator(".probe-comparison-table thead")).toContainText(
      "nats",
    );
    await expect(
      page.locator(".probe-comparison-pagination [aria-live=polite]"),
    ).toHaveText("Page 1 of 1");
    const first = evidence.heads.find((row) => row.causal_paired_examples > 0)!;
    expect(
      Number(
        await page
          .locator(".probe-comparison-table tbody tr td")
          .nth(4)
          .textContent(),
      ),
    ).toBeCloseTo(first.causal_delta_difference!, 5);
    await page.getByLabel("Paired heads only", { exact: true }).uncheck();
    await expect(
      page.locator(".probe-comparison-table tbody tr").first(),
    ).not.toContainText("Unavailable");
    await expect(page.locator(".probe-comparison-table")).toContainText(
      "Unavailable",
    );
    const exportedEvidence = await exported(page);
    expect(exportedEvidence.selected).toEqual(selected);
    expect(exportedEvidence.reference).toEqual(reference);
    expect(exportedEvidence.heads).toEqual(evidence.heads);
    await page.reload();
    await expect(page.locator(".probe-comparison-body")).toBeVisible();
    expect(
      await (await request.get("/api/experiments/" + reference.id)).json(),
    ).toEqual(reference);
    expect((await request.get("/api/runtime")).ok()).toBe(true);
    expect((await (await request.get("/api/runtime")).json()).loaded).toBe(
      false,
    );
    await screenshot(page, "induction-comparison-real-desktop");
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await page.setViewportSize({ width: 390, height: 844 });
    const scrolling = page.getByRole("region", {
      name: "Paired probe head changes",
      exact: true,
    });
    await expect(scrolling).toHaveAttribute("tabindex", "0");
    await scrolling.focus();
    if (
      await scrolling.evaluate((node) => node.scrollWidth > node.clientWidth)
    ) {
      await page.keyboard.press("ArrowRight");
      await expect
        .poll(() => scrolling.evaluate((node) => Math.abs(node.scrollLeft)))
        .toBeGreaterThan(0);
    }
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    expect(errors).toEqual([]);
  } finally {
    if (selectedId) await request.delete("/api/experiments/" + selectedId);
    await request.delete("/api/experiments/" + reference.id);
  }
});

test("comparison failures retry, changing references discards late results and captured sources remain literal", async ({
  page,
  request,
}) => {
  const { a, b } = await pair(request);
  const extra = await seed(request);
  const c = await save(request, extra.id, "Third literal reference");
  let release: (() => void) | undefined;
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  let entered = false;
  let mode: "failed" | "ready" | "delayed" = "failed";
  await page.route("**/api/induction/comparison?**", async (route) => {
    const url = new URL(route.request().url());
    if (mode === "failed")
      return route.fulfill({
        status: 503,
        json: { detail: "Temporary comparison failure" },
      });
    const response = await route.fetch();
    if (mode === "delayed" && url.searchParams.get("reference") === a.id) {
      entered = true;
      await released;
    }
    await route.fulfill({ response });
  });
  try {
    await page.goto("/experiments?experiment=" + b.id + "&compare=" + a.id);
    await expect(page.locator(".probe-comparison .notice.error")).toContainText(
      "Temporary comparison failure",
    );
    await expect(page.locator(".induction-results")).toBeVisible();
    mode = "ready";
    await page
      .getByRole("button", { name: "Retry comparison", exact: true })
      .click();
    await expect(page.locator(".probe-comparison-body")).toBeVisible();
    await page
      .getByLabel("Reference induction probe", { exact: true })
      .selectOption(c.id);
    await expect(page.locator(".probe-comparison-body")).toBeVisible();
    await expect(page.locator(".probe-comparison-sources")).toContainText(
      c.name,
    );
    mode = "delayed";
    await page
      .getByLabel("Reference induction probe", { exact: true })
      .selectOption(a.id);
    await expect.poll(() => entered).toBe(true);
    await page
      .getByLabel("Reference induction probe", { exact: true })
      .selectOption(c.id);
    await expect(page.locator(".probe-comparison-body")).toBeVisible();
    release!();
    await expect(page.locator(".probe-comparison-sources")).toContainText(
      c.name,
    );
    const exportData = await exported(page);
    expect(exportData.reference_record.id).toBe(c.id);
    expect(exportData.reference.name).toBe(c.name);
    expect(exportData.selected).toEqual(b);
    await page
      .getByLabel("Reference induction probe", { exact: true })
      .selectOption("");
    await expect(page.locator(".probe-comparison-body")).toHaveCount(0);
    await expect(page.locator(".induction-results")).toBeVisible();
  } finally {
    release!();
    for (const id of [a.id, b.id, c.id])
      await request.delete("/api/experiments/" + id);
  }
});

test("invalid matching references can be cleared and URL matching works without browser storage", async ({
  page,
  request,
}) => {
  const captured = await seed(request);
  const reference = await save(request, captured.id, "Retained URL reference");
  try {
    await page.goto("/causal/heads/induction?reference=invalid");
    await expect(page.locator(".notice.error")).toContainText(
      "The matching probe reference is invalid.",
    );
    await expect(
      page.getByRole("button", { name: "Run probe", exact: true }),
    ).toHaveCount(0);
    await page
      .getByRole("button", { name: "Clear reference", exact: true })
      .click();
    await expect(
      page.getByRole("button", { name: "Run probe", exact: true }),
    ).toBeEnabled();
    await page.addInitScript(() => {
      Storage.prototype.setItem = () => {
        throw new Error("Storage unavailable");
      };
      Storage.prototype.getItem = () => {
        throw new Error("Storage unavailable");
      };
    });
    await page.goto("/causal/heads/induction?reference=" + reference.id);
    await expect(
      page.getByRole("region", { name: "Matching probe reference" }),
    ).toContainText(reference.name);
    await expect(
      page.getByRole("button", { name: "Run probe", exact: true }),
    ).toBeEnabled();
    await page.getByText("Probe settings", { exact: true }).click();
    await expect(
      page.getByRole("spinbutton", { name: "Tokens per block", exact: true }),
    ).toHaveValue("4");
    await expect(
      page.getByRole("spinbutton", { name: "Tokens per block", exact: true }),
    ).toBeDisabled();
    await page
      .getByRole("button", { name: "Clear reference", exact: true })
      .click();
    await expect(
      page.getByRole("spinbutton", { name: "Tokens per block", exact: true }),
    ).toBeEnabled();
  } finally {
    await request.delete("/api/experiments/" + reference.id);
  }
});

test("checkpoint comparison and matching stay usable in every language, theme and viewport", async ({
  browser,
  request,
  baseURL,
}) => {
  test.setTimeout(240000);
  const { a, b } = await pair(request);
  const catalogues = {
    en: english,
    fa: persian,
    ar: arabic,
    ja: japanese,
    fr: french,
  };
  try {
    for (const [locale, messages] of Object.entries(catalogues)) {
      for (const theme of ["light", "dark"]) {
        const context = await browser.newContext({ baseURL });
        await context.addCookies([
          { name: "lantern_locale", value: locale, url: baseURL! },
          { name: "lantern_theme", value: theme, url: baseURL! },
        ]);
        const page = await context.newPage();
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        try {
          for (const [name, width, height] of [
            ["desktop", 1440, 1100],
            ["mobile", 390, 844],
          ] as const) {
            await page.setViewportSize({ width, height });
            await page.goto(
              "/experiments?experiment=" + b.id + "&compare=" + a.id,
            );
            await expect(page.locator(".probe-comparison-body")).toBeVisible();
            await expect(
              page.getByRole("heading", {
                name: messages["Compare saved probe evidence"],
                exact: true,
              }),
            ).toBeVisible();
            await expect(
              page.locator(".probe-comparison-table thead"),
            ).toContainText(messages["Head"]);
            await expect(
              page.locator(".probe-comparison-table thead"),
            ).toContainText(messages["Change"]);
            if (name === "mobile")
              await expect(page.locator(".probe-table-hint")).toHaveText(
                messages["Scroll horizontally to see all columns."],
              );
            await page.locator(".probe-comparison").scrollIntoViewIfNeeded();
            await screenshot(
              page,
              "probe-comparison-" + locale + "-" + theme + "-" + name,
            );
            expect(
              await page.evaluate(
                () => document.documentElement.scrollWidth <= innerWidth,
              ),
            ).toBe(true);
            if (name === "desktop")
              expect(
                (
                  await new AxeBuilder({ page })
                    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
                    .analyze()
                ).violations,
              ).toEqual([]);
            await page
              .getByRole("button", {
                name: messages["Repeat with another checkpoint"],
                exact: true,
              })
              .click();
            await expect(
              page.getByRole("region", {
                name: messages["Matching probe reference"],
                exact: true,
              }),
            ).toBeVisible();
            await expect(
              page.getByRole("button", {
                name: messages["Run probe"],
                exact: true,
              }),
            ).toBeEnabled();
            await screenshot(
              page,
              "probe-matching-" + locale + "-" + theme + "-" + name,
            );
            expect(
              await page.evaluate(
                () => document.documentElement.scrollWidth <= innerWidth,
              ),
            ).toBe(true);
          }
          await page.setViewportSize({ width: 320, height: 740 });
          await expect(
            page.getByRole("button", {
              name: messages["Clear reference"],
              exact: true,
            }),
          ).toBeVisible();
          expect(
            await page.evaluate(
              () => document.documentElement.scrollWidth <= innerWidth,
            ),
          ).toBe(true);
          expect(errors).toEqual([]);
        } finally {
          await context.close();
        }
      }
    }
  } finally {
    for (const id of [a.id, b.id])
      await request.delete("/api/experiments/" + id);
  }
});

test("clearing a reference during a pending resource check cannot submit an obsolete probe", async ({
  page,
  request,
}) => {
  const captured = await seed(request);
  const reference = await save(request, captured.id, "Pending source");
  let release: (() => void) | undefined;
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  let planned = false,
    submissions = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      request.url().endsWith("/api/induction/probe")
    )
      submissions++;
  });
  await page.route("**/api/induction/plan", async (route) => {
    const response = await route.fetch();
    planned = true;
    await released;
    await route.fulfill({ response });
  });
  try {
    await page.goto("/causal/heads/induction?reference=" + reference.id);
    await page.getByRole("button", { name: "Run probe", exact: true }).click();
    await expect.poll(() => planned).toBe(true);
    await page
      .getByRole("button", { name: "Clear reference", exact: true })
      .click();
    release!();
    await expect(
      page.getByRole("button", { name: "Run probe", exact: true }),
    ).toBeEnabled();
    expect(submissions).toBe(0);
    await expect(
      page.getByText("Probe fits the current resource budget", { exact: true }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("region", { name: "Matching probe reference" }),
    ).toHaveCount(0);
  } finally {
    release!();
    await request.delete("/api/experiments/" + reference.id);
  }
});

test("a model load response arriving after leaving the library cannot redirect the notebook", async ({
  page,
}) => {
  const id = "d".repeat(32);
  let release: (() => void) | undefined;
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  let entered = false;
  await page.route("**/api/models/load", (route) =>
    route.fulfill({
      status: 202,
      json: {
        id,
        kind: "load",
        status: "queued",
        progress: 0,
        message: "Preparing checkpoint",
        model_id: "hf-internal-testing/tiny-random-gpt2",
      },
    }),
  );
  await page.route("**/api/jobs/" + id, async (route) => {
    entered = true;
    await released;
    await route.fulfill({
      json: {
        id,
        kind: "load",
        status: "ready",
        progress: 100,
        message: "Ready",
        model_id: "hf-internal-testing/tiny-random-gpt2",
      },
    });
  });
  try {
    await page.goto("/experiments");
    await page
      .getByRole("link", { name: "Model library", exact: true })
      .click();
    await page
      .getByLabel("Search Hugging Face models")
      .fill("hf-internal-testing/tiny-random-gpt2");
    await page
      .getByRole("button", { name: "Check model", exact: true })
      .click();
    await expect(
      page.getByText("Ready to explore on this server", { exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Download & explore", exact: true })
      .click();
    await expect.poll(() => entered).toBe(true);
    await page.goBack();
    await expect(page).toHaveURL(/experiments/);
    const delivered = page.waitForResponse((response) =>
      response.url().endsWith("/api/jobs/" + id),
    );
    release!();
    await delivered;
    await page.evaluate(
      () =>
        new Promise<void>((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
        ),
    );
    await expect(page).toHaveURL(/experiments/);
  } finally {
    release!();
  }
});

test("matching and clearing a reference preserve the personal probe draft", async ({
  page,
  request,
}) => {
  const captured = await seed(request);
  const reference = await save(
    request,
    captured.id,
    "Reference with different settings",
  );
  try {
    await page.goto("/causal/heads/induction");
    await page.getByText("Probe settings", { exact: true }).click();
    const block = page.getByRole("spinbutton", {
      name: "Tokens per block",
      exact: true,
    });
    const seedInput = page.getByRole("spinbutton", {
      name: "Random seed",
      exact: true,
    });
    await block.fill("7");
    await seedInput.fill("31");
    const runtime = await (await request.get("/api/runtime")).json();
    const key = "lantern-induction-settings-" + runtime.metadata.execution_id;
    await expect
      .poll(() =>
        page.evaluate(
          (key) => JSON.parse(sessionStorage.getItem(key)!).seed,
          key,
        ),
      )
      .toBe(31);
    await page.goto("/experiments?experiment=" + reference.id);
    await page.locator(".probe-comparison summary").click();
    await page
      .getByRole("button", {
        name: "Repeat with another checkpoint",
        exact: true,
      })
      .click();
    await page.getByText("Probe settings", { exact: true }).click();
    await expect(block).toHaveValue("4");
    await expect(seedInput).toHaveValue("19");
    expect(
      await page.evaluate(
        (key) => JSON.parse(sessionStorage.getItem(key)!).seed,
        key,
      ),
    ).toBe(31);
    await page.reload();
    await page.getByText("Probe settings", { exact: true }).click();
    await expect(block).toHaveValue("4");
    await page
      .getByRole("button", { name: "Clear reference", exact: true })
      .click();
    await expect(block).toHaveValue("7");
    await expect(seedInput).toHaveValue("31");
    await expect(block).toBeEnabled();
    expect(
      await (await request.get("/api/experiments/" + reference.id)).json(),
    ).toEqual(reference);
  } finally {
    await request.delete("/api/experiments/" + reference.id);
  }
});
