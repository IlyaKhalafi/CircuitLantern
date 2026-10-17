import AxeBuilder from "@axe-core/playwright";
import {
  expect,
  test as base,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { InductionResult } from "../src/lib/induction";
import type { RuntimeState } from "../src/lib/types";
import english from "../src/i18n/messages/en.json";
import persian from "../src/i18n/messages/fa.json";
import arabic from "../src/i18n/messages/ar.json";
import japanese from "../src/i18n/messages/ja.json";
import french from "../src/i18n/messages/fr.json";

const parameters = {
  block_length: 4,
  repeats: 2,
  examples: 2,
  causal_heads: 1,
  seed: 19,
};
async function load(
  request: APIRequestContext,
  model: string,
  settings: Record<string, unknown> = {},
) {
  const response = await request.post("/api/models/load", {
    data: {
      model_id: model,
      device: "cpu",
      precision: "float32",
      context_length: 128,
      ...settings,
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
const test = base.extend<{ researchSession: void }>({
  researchSession: [
    async ({ request }, use) => {
      const before: RuntimeState = await (
        await request.get("/api/runtime")
      ).json();
      await load(request, "hf-internal-testing/tiny-random-gpt2");
      try {
        await use();
      } finally {
        if (before.loaded && before.metadata) {
          await load(request, before.metadata.id, {
            revision: before.metadata.revision,
            device: before.device,
            precision:
              before.precision || before.metadata.execution_precision || "auto",
            context_length:
              before.context_length ||
              before.metadata.execution_context_length ||
              128,
          });
          const restored: RuntimeState = await (
            await request.get("/api/runtime")
          ).json();
          expect(restored.context_length).toBe(before.context_length);
          expect(restored.precision).toBe(before.precision);
          expect(restored.metadata?.revision).toBe(before.metadata.revision);
        } else
          expect((await request.delete("/api/runtime")).status()).toBe(204);
      }
    },
    { auto: true, timeout: 60000 },
  ],
});

async function seed(request: APIRequestContext) {
  const queued = await request.post("/api/induction/probe", {
    data: parameters,
  });
  expect(queued.status()).toBe(202);
  const job = await queued.json();
  let result: InductionResult | undefined;
  await expect
    .poll(async () => {
      const value = await (await request.get("/api/jobs/" + job.id)).json();
      result = value.result;
      return value.status;
    })
    .toBe("ready");
  return { id: job.id, result: result! };
}
async function expectTokenBlocksWithinCards(page: Page) {
  expect(
    await page.locator(".induction-sequence strong").evaluateAll((nodes) =>
      nodes.every((node) => {
        const range = document.createRange();
        range.selectNodeContents(node);
        const text = range.getBoundingClientRect();
        const card = node.parentElement!.getBoundingClientRect();
        return text.left >= card.left + 2 && text.right <= card.right - 2;
      }),
    ),
  ).toBe(true);
}

async function screenshot(page: Page, name: string) {
  const directory = join(process.cwd(), "../artifacts/screenshots");
  mkdirSync(directory, { recursive: true });
  await page.evaluate(async () => {
    await document.fonts.ready;
  });
  await page.screenshot({
    path: join(directory, name + ".png"),
    fullPage: false,
  });
}

test("plan and run real repeated token probes, inspect heads and retain unloaded notebook evidence", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let savedId = "";
  try {
    await page.goto("/causal/heads");
    await page.getByRole("link", { name: "Explore induction heads" }).click();
    await expect(page).toHaveTitle("Induction probe · Circuit Lantern");
    await expect(
      page.getByRole("link", { name: "Head ablation", exact: true }),
    ).toHaveAttribute("href", "/causal/heads");
    await expect(
      page.getByRole("link", { name: "Research lab", exact: true }),
    ).toHaveClass(/active/);
    await expect(
      page.getByRole("button", { name: "Run probe", exact: true }),
    ).toBeEnabled();
    await page.getByText("Probe settings", { exact: true }).click();
    for (const [label, value] of [
      ["Tokens per block", 4],
      ["Repeats", 2],
      ["Probe examples", 2],
      ["Candidate heads to ablate", 1],
      ["Random seed", 19],
    ] as const)
      await page
        .getByRole("spinbutton", { name: label, exact: true })
        .fill(String(value));
    let submissions = 0;
    page.on("request", (request) => {
      if (
        request.url().endsWith("/api/induction/probe") &&
        request.method() === "POST"
      )
        submissions++;
    });
    await page
      .getByRole("button", { name: "Check resources", exact: true })
      .click();
    await expect(
      page.getByText("Probe fits the current resource budget", { exact: true }),
    ).toBeVisible();
    expect(submissions).toBe(0);
    await expect(page.locator(".notice.error")).toHaveCount(0);
    await page.getByRole("button", { name: "Run probe", exact: true }).click();
    await expect(
      page.locator(".trace-meta").getByText("Probe complete", { exact: true }),
    ).toBeVisible();
    expect(submissions).toBe(1);
    await expect(page.locator(".induction-rank-list button")).toHaveCount(12);
    await page
      .locator(".induction-pagination")
      .getByRole("button", { name: "Next", exact: true })
      .click();
    await expect(page.locator(".induction-rank-list button")).toHaveCount(8);
    await page.getByLabel("Probe results layer").selectOption("2");
    await expect(page.locator(".induction-rank-list button")).toHaveCount(4);
    await page.getByLabel("Probe results layer").selectOption("all");
    await page.locator(".induction-rank-list button").first().click();
    const download = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export probe", exact: true })
      .click();
    const stream = await (await download).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    const captured: InductionResult = JSON.parse(
      Buffer.concat(chunks).toString(),
    );
    expect(captured.complete).toBe(true);
    expect(captured.heads).toHaveLength(20);
    expect(captured.completed_forwards).toBe(4);
    expect(captured.dataset.probes[0].token_ids).toHaveLength(9);
    expect(captured.dataset.probes[0].repeated_queries).toEqual([5, 6, 7, 8]);
    expect(captured.requested_causal_heads).toBe(1);
    expect(captured.dataset_fingerprint).toMatch(/^[a-f0-9]{64}$/);
    const causal = captured.causal_results[0];
    const delta = page.locator(".induction-causal dd").first();
    expect(Number(await delta.textContent())).toBeCloseTo(
      causal.mean_delta_log_probability,
      5,
    );
    await page.locator(".induction-rank-list button").nth(1).click();
    await expect(
      page.getByText(
        "This head has no completed ablation pairs. Attention scores remain pattern evidence.",
        { exact: true },
      ),
    ).toBeVisible();
    await page.locator(".induction-rank-list button").first().click();
    await screenshot(page, "induction-real-desktop");
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await page.reload();
    await expect(
      page.locator(".trace-meta").getByText("Probe complete", { exact: true }),
    ).toBeVisible();
    await page.getByText("Probe settings", { exact: true }).click();
    await expect(
      page.getByRole("spinbutton", { name: "Tokens per block", exact: true }),
    ).toHaveValue("4");
    await page
      .getByLabel("Experiment name", { exact: true })
      .fill("Repeated token evidence");
    const saved = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/experiments/from-job") &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    savedId = (await (await saved).json()).id;
    expect((await request.delete("/api/runtime")).status()).toBe(204);
    await page.getByRole("link", { name: "View saved experiment" }).click();
    await expect(page.locator(".induction-results")).toBeVisible();
    await expect(
      page.locator(".trace-meta").getByText("Probe complete", { exact: true }),
    ).toBeVisible();
    const record = await (
      await request.get("/api/experiments/" + savedId)
    ).json();
    expect(record.kind).toBe("induction_probe");
    expect(record.result).toEqual(captured);
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await screenshot(page, "induction-notebook-mobile");
    await page
      .getByRole("link", { name: "All experiments", exact: true })
      .click();
    await page
      .getByRole("combobox", { name: "Experiment type", exact: true })
      .selectOption("induction_probe");
    await expect(
      page.getByRole("heading", {
        name: "Repeated token evidence",
        exact: true,
      }),
    ).toBeVisible();
    expect(errors).toEqual([]);
  } finally {
    if (savedId) await request.delete("/api/experiments/" + savedId);
  }
});

test("restore a running probe and retain partial evidence after cancellation", async ({
  page,
  request,
}) => {
  const captured = await seed(request);
  const fakeId = "e".repeat(32);
  let cancelled = false;
  const partial: InductionResult = {
    ...captured.result,
    complete: false,
    cancelled: true,
    measured_examples: 1,
    completed_causal_heads: 0,
    causal_results: [],
    probe_metrics: captured.result.probe_metrics.slice(0, 1),
  };
  await page.route("**/api/jobs/" + fakeId, (route) =>
    route.fulfill({
      json: {
        id: fakeId,
        kind: "induction_probe",
        status: cancelled ? "cancelled" : "running",
        model_id: captured.result.model_id,
        message: cancelled
          ? "Induction probe cancelled. Completed measurements are retained."
          : "Scoring repeated token attention",
        progress: 25,
        result: cancelled ? partial : undefined,
      },
    }),
  );
  await page.route("**/api/jobs/" + fakeId + "/cancel", (route) => {
    cancelled = true;
    return route.fulfill({
      json: { status: "running", cancel_requested: true },
    });
  });
  await page.addInitScript(
    (id) => sessionStorage.setItem("lantern-induction-probe-job", id),
    fakeId,
  );
  await page.goto("/causal/heads/induction");
  await expect(
    page.getByRole("button", { name: "Cancel probe", exact: true }),
  ).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Cancel probe", exact: true }).click();
  await expect(
    page.locator(".trace-meta").getByText("Partial probe", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Cancelled between forwards", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".induction-rank-list button")).toHaveCount(12);
  await expect(
    page.getByText(
      "This head has no completed ablation pairs. Attention scores remain pattern evidence.",
      { exact: true },
    ),
  ).toBeVisible();
  await screenshot(page, "induction-cancelled-desktop");
});

test("probe setup and captured results work in every language, theme and viewport", async ({
  browser,
  request,
  baseURL,
}) => {
  test.setTimeout(180000);
  const captured = await seed(request);
  const catalogues = {
    en: english,
    fa: persian,
    ar: arabic,
    ja: japanese,
    fr: french,
  };
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
          await page.goto("/causal/heads/induction");
          await expect(
            page.getByRole("button", {
              name: messages["Run probe"],
              exact: true,
            }),
          ).toBeEnabled();
          await expect(page).toHaveTitle(
            messages["Induction probe"] + " · Circuit Lantern",
          );
          await expectTokenBlocksWithinCards(page);
          await screenshot(
            page,
            "induction-setup-" + locale + "-" + theme + "-" + name,
          );
          expect(
            await page.evaluate(
              () => document.documentElement.scrollWidth <= innerWidth,
            ),
          ).toBe(true);
          await page.evaluate(
            (id) => sessionStorage.setItem("lantern-induction-probe-job", id),
            captured.id,
          );
          await page.reload();
          await expect(page.locator(".induction-results")).toBeVisible();
          await page.locator(".induction-results").scrollIntoViewIfNeeded();
          await screenshot(
            page,
            "induction-results-" + locale + "-" + theme + "-" + name,
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
          await page.evaluate(() =>
            sessionStorage.removeItem("lantern-induction-probe-job"),
          );
        }
        await page.setViewportSize({ width: 320, height: 740 });
        await expectTokenBlocksWithinCards(page);
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
});

test("resource and capability failures recover without submitting a probe", async ({
  page,
}) => {
  let failed = true;
  let submissions = 0;
  page.on("request", (request) => {
    if (
      request.url().endsWith("/api/induction/probe") &&
      request.method() === "POST"
    )
      submissions++;
  });
  await page.route("**/api/induction/capabilities", async (route) => {
    if (failed)
      return route.fulfill({
        status: 503,
        json: { detail: "Temporary probe service failure" },
      });
    return route.continue();
  });
  await page.goto("/causal/heads/induction");
  await expect(page.locator(".notice.error")).toContainText(
    "Temporary probe service failure",
  );
  failed = false;
  await page
    .getByRole("button", { name: "Refresh runtime", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Run probe", exact: true }),
  ).toBeEnabled();
  await page.route("**/api/induction/plan", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      json: {
        ...(await response.json()),
        compatible: false,
        blockers: [
          "Insufficient available RAM for probe baselines and attention measurements.",
        ],
      },
    });
  });
  await page.getByRole("button", { name: "Run probe", exact: true }).click();
  await expect(
    page.getByText("This probe needs fewer resources", { exact: true }),
  ).toBeVisible();
  expect(submissions).toBe(0);
  await page.getByText("Probe settings", { exact: true }).click();
  await page
    .getByRole("spinbutton", { name: "Probe examples", exact: true })
    .fill("");
  await page
    .getByRole("button", { name: "Check resources", exact: true })
    .click();
  expect(submissions).toBe(0);
});

test("invalid collapsed probe settings reopen the control and reload preserves other valid fields", async ({
  page,
  request,
}) => {
  const caps = await (await request.get("/api/induction/capabilities")).json();
  const execution = caps.execution.execution_id;
  await page.addInitScript(
    ({ execution, value }) => {
      if (!sessionStorage.getItem("lantern-induction-settings-" + execution))
        sessionStorage.setItem(
          "lantern-induction-settings-" + execution,
          JSON.stringify(value),
        );
    },
    {
      execution,
      value: {
        ...caps.recommended,
        examples: null,
        seed: 23,
        legacy_setting: 1,
      },
    },
  );
  let submissions = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      request.url().endsWith("/api/induction/probe")
    )
      submissions++;
  });
  await page.goto("/causal/heads/induction");
  await expect(
    page.getByRole("button", { name: "Run probe", exact: true }),
  ).toBeEnabled();
  await page.getByText("Probe settings", { exact: true }).click();
  const example = page.getByRole("spinbutton", {
    name: "Probe examples",
    exact: true,
  });
  const seedInput = page.getByRole("spinbutton", {
    name: "Random seed",
    exact: true,
  });
  await expect(seedInput).toHaveValue("23");
  await expect(example).toHaveValue(String(caps.recommended.examples));
  await example.fill("");
  await expect(page.locator(".induction-form")).not.toContainText("NaN");
  await page.getByText("Probe settings", { exact: true }).click();
  await page.getByRole("button", { name: "Run probe", exact: true }).click();
  await expect(page.locator(".induction-advanced")).toHaveAttribute("open", "");
  await expect(example).toBeFocused();
  expect(submissions).toBe(0);
  await example.fill("999");
  await expect(page.locator(".induction-form")).toContainText(
    "Enter valid probe settings to estimate the workload.",
  );
  await page.getByText("Probe settings", { exact: true }).click();
  await page
    .getByRole("button", { name: "Check resources", exact: true })
    .click();
  await expect(page.locator(".induction-advanced")).toHaveAttribute("open", "");
  await expect(example).toBeFocused();
  expect(submissions).toBe(0);
  await page.reload();
  await page.getByText("Probe settings", { exact: true }).click();
  await expect(seedInput).toHaveValue("23");
  await expect(example).toHaveValue(String(caps.recommended.examples));
  const planned = page.waitForRequest((request) =>
    request.url().endsWith("/api/induction/plan"),
  );
  await page
    .getByRole("button", { name: "Check resources", exact: true })
    .click();
  const payload = (await planned).postDataJSON();
  expect(payload.seed).toBe(23);
  expect(payload).not.toHaveProperty("legacy_setting");
  expect(submissions).toBe(0);
});
