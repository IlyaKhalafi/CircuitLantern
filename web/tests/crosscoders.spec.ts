import AxeBuilder from "@axe-core/playwright";
import {
  expect,
  test as base,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import type { RuntimeState } from "../src/lib/types";
import type { CaptureSummary, CrosscoderSummary } from "../src/lib/crosscoders";

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
async function load(
  request: APIRequestContext,
  model: string,
  settings: {
    revision?: string;
    device?: string;
    precision?: string;
    context_length?: number;
  } = {},
) {
  const response = await request.post("/api/models/load", {
    data: {
      model_id: model,
      device: "cpu",
      precision: "float32",
      context_length: 256,
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
        } else {
          expect((await request.delete("/api/runtime")).status()).toBe(204);
        }
      }
    },
    { auto: true, timeout: 60000 },
  ],
});

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
test("retain paired checkpoint activations, train unloaded and explore joint feature evidence", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    sessionStorage.setItem(
      "lantern-crosscoder-training-draft",
      JSON.stringify({ device: "cuda" }),
    );
  });
  await page.route("**/api/crosscoders/capabilities", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      json: { ...(await response.json()), devices: ["cpu"] },
    });
  });
  let comparisonId = "";
  let referenceId = "";
  let comparedId = "";
  try {
    await page.goto("/features");
    await page
      .getByRole("link", { name: "Compare shared features", exact: true })
      .click();
    await expect(page).toHaveURL(/\/features\/compare$/);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(
      "Shared feature comparisons",
    );
    await expect(page.locator(".breadcrumb strong")).toHaveText(
      "Shared feature comparisons",
    );
    await expect(page.locator(".sidebar nav a.active")).toHaveAttribute(
      "href",
      "/playground",
    );
    const execution = page.locator(".crosscoder-session .execution-context");
    await expect(execution).toContainText("FP32");
    await expect(execution).toContainText("CPU");
    await expect(execution).toContainText("256 context tokens");
    const reference = page.locator("section.panel").filter({
      has: page.getByRole("heading", {
        name: "1 · Reference activations",
        exact: true,
      }),
    });
    await reference
      .getByLabel("Capture name", { exact: true })
      .fill("Reference {raw}");
    await reference
      .getByLabel("Training corpus", { exact: true })
      .fill(documents);
    await reference
      .getByText("Advanced capture settings", { exact: true })
      .click();
    await reference
      .getByLabel("Maximum token positions", { exact: true })
      .fill("");
    await expect(
      reference.getByRole("button", { name: "Capture reference", exact: true }),
    ).toBeDisabled();
    await reference
      .getByLabel("Maximum token positions", { exact: true })
      .fill("128");
    await reference
      .getByRole("button", { name: "Check resources", exact: true })
      .click();
    await expect(
      reference.getByText("Resources available", { exact: true }),
    ).toBeVisible();
    await page.evaluate(() => scrollTo(0, 0));
    await screenshot(page, "crosscoder-reference-light");
    await reference
      .getByRole("button", { name: "Capture reference", exact: true })
      .click();
    const matched = page.locator("section.panel").filter({
      has: page.getByRole("heading", {
        name: "2 · Matched activations",
        exact: true,
      }),
    });
    await expect(matched).toBeVisible();
    const captures: CaptureSummary[] = await (
      await request.get("/api/crosscoders/captures")
    ).json();
    referenceId = captures.find(
      (value) => value.name === "Reference {raw}",
    )!.id;
    await expect(
      matched.getByText("The reference checkpoint is still loaded.", {
        exact: false,
      }),
    ).toBeVisible();
    await page
      .locator(".crosscoder-session")
      .getByRole("link", { name: "Open model library", exact: true })
      .click();
    await load(request, "circuit-lantern-test/finetuned");
    await page.goto("/features/compare");
    await expect(
      reference.getByText("Reference {raw}", { exact: true }),
    ).toBeVisible();
    await expect(
      page.locator(".crosscoder-session .execution-context bdi").first(),
    ).toHaveText("circuit-lantern-test/finetuned");
    await matched
      .getByLabel("Capture name", { exact: true })
      .fill("Compared {raw}");
    await matched.getByRole("heading").evaluate((element) =>
      window.scrollTo({
        top: element.getBoundingClientRect().top + scrollY - 24,
        behavior: "instant",
      }),
    );
    await screenshot(page, "crosscoder-matched-light");
    await matched
      .getByRole("button", { name: "Capture matched activations", exact: true })
      .click();
    const training = page.locator("section.panel").filter({
      has: page.getByRole("heading", {
        name: "3 · Train shared features",
        exact: true,
      }),
    });
    await expect(training).toBeVisible();
    await expect(
      training.getByLabel("Training device", { exact: true }),
    ).toHaveValue("cuda");
    await expect(training.getByRole("status")).toHaveText(
      "The saved training device is unavailable on this server. Choose an available device before training.",
    );
    await expect(
      training.getByRole("button", {
        name: "Train shared features",
        exact: true,
      }),
    ).toBeDisabled();
    await training
      .getByLabel("Training device", { exact: true })
      .selectOption("cpu");
    await expect(training.getByRole("status")).toHaveCount(0);
    const retained: CaptureSummary[] = await (
      await request.get("/api/crosscoders/captures")
    ).json();
    comparedId = retained.find((value) => value.name === "Compared {raw}")!.id;
    expect(
      retained.find((value) => value.id === comparedId)!.matched_to!.id,
    ).toBe(referenceId);
    await request.delete("/api/runtime");
    await page.getByRole("button", { name: "Refresh", exact: true }).click();
    await expect(
      page
        .locator(".crosscoder-session")
        .getByText("No model loaded", { exact: true }),
    ).toBeVisible();
    await training
      .getByLabel("Comparison name", { exact: true })
      .fill("Joint comparison {raw}");
    await training
      .getByText("Advanced training settings", { exact: true })
      .click();
    await training.getByLabel("Optimizer steps", { exact: true }).fill("16");
    await training.getByLabel("Expansion factor", { exact: true }).fill("1");
    await training
      .getByRole("button", { name: "Check resources", exact: true })
      .click();
    await expect(
      training.getByText("Resources available", { exact: true }),
    ).toBeVisible();
    await training.getByRole("heading").evaluate((element) =>
      window.scrollTo({
        top: element.getBoundingClientRect().top + scrollY - 24,
        behavior: "instant",
      }),
    );
    await screenshot(page, "crosscoder-training-light");
    await training
      .getByRole("button", { name: "Train shared features", exact: true })
      .click();
    await expect(page.locator(".crosscoder-inspector")).toBeVisible();
    const comparisons: CrosscoderSummary[] = await (
      await request.get("/api/crosscoders")
    ).json();
    comparisonId = comparisons.find(
      (value) => value.name === "Joint comparison {raw}",
    )!.id;
    const record = comparisons.find((value) => value.id === comparisonId)!;
    expect(record.banks[0].source.model_id).toBe(
      "hf-internal-testing/tiny-random-gpt2",
    );
    expect(record.banks[1].source.model_id).toBe(
      "circuit-lantern-test/finetuned",
    );
    expect(record.training_summary.completed_steps).toBe(16);
    expect((await (await request.get("/api/runtime")).json()).loaded).toBe(
      false,
    );
    const inspector = page.locator(".crosscoder-inspector");
    await expect(
      inspector.getByLabel("Feature label", { exact: true }),
    ).toBeVisible();
    await inspector
      .getByLabel("Feature label", { exact: true })
      .fill("Shared {literal}");
    await inspector
      .getByLabel("Research notes", { exact: true })
      .fill("Observed joint context");
    await inspector
      .getByRole("button", { name: "Save annotation", exact: true })
      .click();
    await expect(
      inspector.getByRole("button", { name: "Save annotation", exact: true }),
    ).toBeDisabled();
    await page
      .getByLabel("Search features", { exact: true })
      .fill("Shared {literal}");
    await expect(page.locator(".crosscoder-feature-row")).toHaveCount(1);
    await expect(page.locator(".crosscoder-feature-row")).toContainText(
      "Shared {literal}",
    );
    const download = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export evidence", exact: true })
      .click();
    const exported = await download;
    expect(exported.suggestedFilename()).toBe(
      "crosscoder-" + comparisonId + ".json",
    );
    let downloadFailure: { status: number; message: string } | null = null;
    await page.route(
      "**/api/crosscoders/" + comparisonId + "/download",
      (route) => {
        if (downloadFailure)
          return route.fulfill({
            status: downloadFailure.status,
            json: { detail: downloadFailure.message },
          });
        return route.continue();
      },
    );
    for (const [status, message] of [
      [400, "The saved tensor artifact checksum does not match."],
      [404, "The saved tensor file was not found."],
      [409, "Downloads are busy. Try again shortly."],
    ] as const) {
      downloadFailure = { status, message };
      await page
        .getByRole("button", { name: "Download weights", exact: true })
        .click();
      await expect(
        page.locator(".notice.error").filter({ hasText: message }),
      ).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Download weights", exact: true }),
      ).toBeEnabled();
      if (status === 400) {
        await page
          .locator(".notice.error")
          .filter({ hasText: message })
          .scrollIntoViewIfNeeded();
        await screenshot(page, "crosscoder-download-integrity-error");
      }
    }
    downloadFailure = null;
    const weights = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Download weights", exact: true })
      .click();
    const downloadedWeights = await weights;
    expect(downloadedWeights.suggestedFilename()).toContain(comparisonId);
    expect(await downloadedWeights.failure()).toBeNull();
    const weightStream = await downloadedWeights.createReadStream();
    const digest = createHash("sha256");
    let receivedBytes = 0;
    for await (const chunk of weightStream!) {
      digest.update(chunk);
      receivedBytes += chunk.length;
    }
    expect(digest.digest("hex")).toBe(record.tensors_sha256);
    const proxyPrepared = await request.post(
      "/api/crosscoders/" + comparisonId + "/download",
    );
    expect(proxyPrepared.status()).toBe(200);
    const preparedStream = await proxyPrepared.json();
    const proxyDownload = await request.get(
      "/api/crosscoders/downloads/" + preparedStream.ticket,
    );
    expect(proxyDownload.status()).toBe(200);
    expect(proxyDownload.headers()["content-length"]).toBe(
      String(receivedBytes),
    );
    expect((await proxyDownload.body()).length).toBe(preparedStream.size);
    expect(
      createHash("sha256")
        .update(await proxyDownload.body())
        .digest("hex"),
    ).toBe(record.tensors_sha256);

    let releasePreparation = () => {};
    let sawPreparation = () => {};
    const preparationGate = new Promise<void>((resolve) => {
      releasePreparation = resolve;
    });
    const preparationSeen = new Promise<void>((resolve) => {
      sawPreparation = resolve;
    });
    let nativeRequests = 0;
    const countNative = (request: import("@playwright/test").Request) => {
      if (
        new URL(request.url()).pathname.startsWith(
          "/api/crosscoders/downloads/",
        )
      )
        nativeRequests++;
    };
    page.on("request", countNative);
    await page.route(
      "**/api/crosscoders/" + comparisonId + "/download",
      async (route) => {
        sawPreparation();
        await preparationGate;
        await route.fulfill({ json: { ticket: "e".repeat(64) } });
      },
    );
    try {
      await page
        .getByRole("button", { name: "Download weights", exact: true })
        .click();
      await preparationSeen;
      await expect(
        page.getByRole("button", { name: "Checking download", exact: true }),
      ).toBeDisabled();
      await page
        .getByLabel("Choose a comparison", { exact: true })
        .selectOption("");
      releasePreparation();
      await page
        .getByLabel("Choose a comparison", { exact: true })
        .selectOption(comparisonId);
      await expect(page.locator(".crosscoder-inspector")).toBeVisible();
      await page.evaluate(
        () =>
          new Promise<void>((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
          ),
      );
      expect(nativeRequests).toBe(0);
    } finally {
      releasePreparation();
      page.off("request", countNative);
    }
    await page.reload();
    await expect(page.locator(".crosscoder-inspector")).toBeVisible();
    await expect(
      page.getByRole("heading", {
        name: "Joint comparison {raw}",
        exact: true,
      }),
    ).toBeVisible();
    await page
      .getByText("Training diagnostics and provenance", { exact: true })
      .click();
    await expect(page.locator(".recharts-wrapper")).toBeVisible();
    const violations = (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations;
    expect(violations).toEqual([]);
    for (const theme of ["light", "dark"]) {
      await page
        .locator(".theme-select button")
        .nth(theme === "light" ? 0 : 1)
        .click();
      await page.evaluate(() => scrollTo(0, 0));
      await screenshot(page, "crosscoder-overview-" + theme);
      await page.locator(".crosscoder-inspector h2").evaluate((element) => {
        window.scrollTo({
          top:
            element.getBoundingClientRect().top +
            scrollY -
            (innerWidth <= 720 ? 112 : 24),
          behavior: "instant",
        });
      });
      await screenshot(page, "crosscoder-feature-" + theme);
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator(".crosscoder-feature-row").first().click();
    await expect
      .poll(() =>
        page.evaluate(() =>
          Boolean(document.activeElement?.closest(".crosscoder-inspector")),
        ),
      )
      .toBe(true);
    await page.setViewportSize({ width: 1440, height: 1100 });
    for (const locale of ["fa", "ar", "ja", "fr", "en"]) {
      await page.locator(".language-select select").selectOption(locale);
      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      for (const theme of ["light", "dark"]) {
        await page
          .locator(".theme-select button")
          .nth(theme === "light" ? 0 : 1)
          .click();
        await page.setViewportSize({ width: 390, height: 844 });
        await page.evaluate(() => scrollTo(0, 0));
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBe(true);
        await screenshot(
          page,
          "crosscoder-overview-" + locale + "-" + theme + "-mobile",
        );
        await page.locator(".crosscoder-inspector h2").evaluate((element) => {
          window.scrollTo({
            top:
              element.getBoundingClientRect().top +
              scrollY -
              (innerWidth <= 720 ? 112 : 24),
            behavior: "instant",
          });
        });
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBe(true);
        await screenshot(
          page,
          "crosscoder-feature-" + locale + "-" + theme + "-mobile",
        );
        await page.setViewportSize({ width: 1440, height: 1100 });
      }
      await expect(
        page.getByRole("heading", {
          name: "Joint comparison {raw}",
          exact: true,
        }),
      ).toBeVisible();
      const legend = page.locator(".crosscoder-norm-key > span");
      expect((await legend.nth(0).boundingBox())!.x).toBeLessThan(
        (await legend.nth(1).boundingBox())!.x,
      );
    }

    await page.getByText("Manage retained data", { exact: true }).click();
    const storage = page.locator(".crosscoder-storage");
    await storage
      .getByLabel("Saved artifact", { exact: true })
      .selectOption(referenceId);
    const captureTensors = page.waitForEvent("download");
    await storage
      .getByRole("button", { name: "Download tensors", exact: true })
      .click();
    const downloadedCapture = await captureTensors;
    expect(await downloadedCapture.failure()).toBeNull();
    expect(downloadedCapture.suggestedFilename()).toContain(referenceId);
    const tensorStream = await downloadedCapture.createReadStream();
    const captureDigest = createHash("sha256");
    for await (const chunk of tensorStream!) captureDigest.update(chunk);
    expect(captureDigest.digest("hex")).toBe(
      captures.find((item) => item.id === referenceId)!.tensors_sha256,
    );
    const captureExport = page.waitForEvent("download");
    await storage
      .getByRole("button", { name: "Export evidence", exact: true })
      .click();
    expect((await captureExport).suggestedFilename()).toBe(
      "capture-" + referenceId + ".json",
    );
    await storage
      .getByRole("button", { name: "Delete saved artifact", exact: true })
      .click();
    await storage.getByRole("button", { name: "Cancel", exact: true }).click();
    expect(
      (await request.get("/api/crosscoders/captures/" + referenceId)).status(),
    ).toBe(200);
    await storage
      .getByRole("button", { name: "Delete saved artifact", exact: true })
      .click();
    await storage
      .getByRole("button", { name: "Confirm deletion", exact: true })
      .click();
    await expect(
      storage.getByLabel("Saved artifact", { exact: true }),
    ).toHaveValue("");
    expect(
      (
        await request.get("/api/crosscoders/" + comparisonId + "/overview")
      ).status(),
    ).toBe(200);
    await storage
      .getByLabel("Artifact type", { exact: true })
      .selectOption("crosscoder");
    await storage
      .getByLabel("Saved artifact", { exact: true })
      .selectOption(comparisonId);
    await storage
      .getByRole("button", { name: "Delete saved artifact", exact: true })
      .click();
    await storage
      .getByRole("button", { name: "Confirm deletion", exact: true })
      .click();
    await expect(
      page.getByRole("heading", {
        name: "No shared feature comparison selected",
        exact: true,
      }),
    ).toBeVisible();
    expect(errors).toEqual([]);
  } finally {
    if (comparisonId) await request.delete("/api/crosscoders/" + comparisonId);
    for (const id of [referenceId, comparedId])
      if (id) await request.delete("/api/crosscoders/captures/" + id);
  }
});

test("a capture job survives reload and stops without publishing incomplete tensors", async ({
  page,
}) => {
  const identifier = "d".repeat(32);
  let cancelled = false;
  let requests = 0;
  await page.route("**/api/crosscoders/captures", async (route) => {
    if (route.request().method() !== "POST") return route.continue();
    requests++;
    return route.fulfill({
      status: 202,
      json: {
        id: identifier,
        model_id: "fixture",
        status: "queued",
        progress: 0,
        message: "Collecting aligned residual activations",
      },
    });
  });
  await page.route("**/api/jobs/" + identifier, (route) =>
    route.fulfill({
      json: {
        id: identifier,
        model_id: "fixture",
        status: cancelled ? "cancelled" : "running",
        progress: 20,
        message: cancelled
          ? "Residual capture cancelled. Completed captures are retained."
          : "Collecting aligned residual activations",
        ...(cancelled ? { result: { captures: [], complete: false } } : {}),
      },
    }),
  );
  await page.route("**/api/jobs/" + identifier + "/cancel", (route) => {
    cancelled = true;
    return route.fulfill({
      json: { status: "running", cancel_requested: true },
    });
  });
  await page.goto("/features/compare");
  const capture = page.getByRole("button", {
    name: "Capture reference",
    exact: true,
  });
  await expect(capture).toBeEnabled();
  await capture.click();
  await expect(capture).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeVisible();
  await expect(capture).toBeDisabled();
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByText(
      "Residual capture cancelled. Completed captures are retained.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(capture).toBeEnabled();
  expect(requests).toBe(1);
  await expect(
    page.getByRole("heading", { name: "2 · Matched activations", exact: true }),
  ).toHaveCount(0);
});

test("capture preparation remains readable in all locales and themes", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/features/compare");
  await expect(page.locator(".crosscoder-session")).toBeVisible();
  for (const locale of ["en", "fa", "ar", "ja", "fr"]) {
    await page.locator(".language-select select").selectOption(locale);
    await expect(page.locator("html")).toHaveAttribute("lang", locale);
    expect(
      await page
        .locator(".crosscoder-workbench textarea")
        .first()
        .evaluate((element) => getComputedStyle(element).direction),
    ).toBe("ltr");

    for (const theme of ["light", "dark"]) {
      await page
        .locator(".theme-select button")
        .nth(theme === "light" ? 0 : 1)
        .click();
      for (const [name, width, height] of [
        ["desktop", 1440, 1100],
        ["mobile", 390, 844],
      ] as const) {
        await page.setViewportSize({ width, height });
        await page.evaluate(() => scrollTo(0, 0));
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBe(true);
        await screenshot(
          page,
          "crosscoder-prepare-" + locale + "-" + theme + "-" + name,
        );
      }
    }
    await page.setViewportSize({ width: 320, height: 844 });
    const title = page.locator(".breadcrumb strong");
    expect(
      await title.evaluate(
        (element) => element.scrollWidth <= element.clientWidth,
      ),
    ).toBe(true);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.setViewportSize({ width: 1440, height: 1100 });
  }
  expect(errors).toEqual([]);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
});

test("a delayed annotation cannot overwrite another shared feature draft", async ({
  page,
  request,
}) => {
  const captures: string[] = [];
  let comparison = "";
  let release = () => {};
  async function run(path: string, data: Record<string, unknown>) {
    const response = await request.post(path, { data });
    expect(response.status()).toBe(202);
    const started = await response.json();
    let finished: {
      status: string;
      result?: { captures: CaptureSummary[]; crosscoders: CrosscoderSummary[] };
    };
    await expect
      .poll(async () => {
        finished = await (await request.get("/api/jobs/" + started.id)).json();
        return finished.status;
      })
      .toBe("ready");
    return finished!.result!;
  }
  try {
    const reference = (
      await run("/api/crosscoders/captures", {
        name: "Race reference",
        documents: documents.split("\n"),
        layer: 0,
        max_samples: 64,
      })
    ).captures[0];
    captures.push(reference.id);
    const compared = (
      await run("/api/crosscoders/captures/" + reference.id + "/match", {
        name: "Race compared",
        layer: 0,
      })
    ).captures[0];
    captures.push(compared.id);
    comparison = (
      await run("/api/crosscoders/train", {
        name: "Race comparison",
        capture_ids: captures,
        expansion: 1,
        steps: 4,
      })
    ).crosscoders[0].id;
    let seen = () => {};
    const intercepted = new Promise<void>((resolve) => {
      seen = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let completed = () => {};
    const reply = new Promise<void>((resolve) => {
      completed = resolve;
    });
    await page.route(
      "**/api/crosscoders/" + comparison + "/features/0",
      async (route) => {
        if (route.request().method() !== "PATCH") return route.continue();
        seen();
        await gate;
        const response = await route.fetch();
        await route.fulfill({ response });
        completed();
      },
    );
    let failOverview = true;
    await page.route(
      "**/api/crosscoders/" + comparison + "/overview",
      (route) => {
        if (failOverview) {
          failOverview = false;
          return route.fulfill({
            status: 503,
            json: { detail: "Transient overview failure" },
          });
        }
        return route.continue();
      },
    );
    await page.goto("/features/compare");
    await page
      .getByRole("button", { name: "Explore comparisons", exact: true })
      .click();
    await page
      .getByLabel("Choose a comparison", { exact: true })
      .selectOption(comparison);
    await expect(
      page.getByRole("alert").filter({ hasText: "Transient overview failure" }),
    ).toContainText("Transient overview failure");
    await page.getByRole("button", { name: "Refresh", exact: true }).click();
    await expect(page.locator(".crosscoder-inspector")).toBeVisible();
    const search = page.getByLabel("Search features", { exact: true });
    await search.fill("0");
    await page
      .locator(".crosscoder-feature-row")
      .filter({ has: page.getByText("Feature 0", { exact: true }) })
      .click();
    const inspector = page.locator(".crosscoder-inspector");
    await inspector
      .getByLabel("Feature label", { exact: true })
      .fill("Delayed label");
    await inspector
      .getByRole("button", { name: "Save annotation", exact: true })
      .click();
    await intercepted;
    await search.fill("1");
    await page
      .locator(".crosscoder-feature-row")
      .filter({ has: page.getByText("Feature 1", { exact: true }) })
      .click();
    await inspector
      .getByLabel("Research notes", { exact: true })
      .fill("Keep this newer draft");
    release();
    await reply;
    await expect(
      inspector.getByRole("heading", { name: "Feature 1", exact: true }),
    ).toBeVisible();
    await expect(
      inspector.getByLabel("Research notes", { exact: true }),
    ).toHaveValue("Keep this newer draft");
    await expect(
      inspector.getByLabel("Feature label", { exact: true }),
    ).toHaveValue("");
    await expect(
      inspector.getByRole("button", { name: "Save annotation", exact: true }),
    ).toBeEnabled();
    await page.getByRole("button", { name: "Refresh", exact: true }).click();
    await expect(
      inspector.getByLabel("Research notes", { exact: true }),
    ).toHaveValue("Keep this newer draft");

    expect(
      (
        await (
          await request.get("/api/crosscoders/" + comparison + "/features/0")
        ).json()
      ).label,
    ).toBe("Delayed label");
  } finally {
    release();
    if (comparison) await request.delete("/api/crosscoders/" + comparison);
    for (const identifier of captures)
      await request.delete("/api/crosscoders/captures/" + identifier);
  }
});

test("damaged summaries stay outside research selection and can be removed from retained data", async ({
  page,
}) => {
  const identifier = "f".repeat(32);
  let removed = false;
  const issue = {
    id: identifier,
    kind: "capture",
    name: identifier,
    created_at: "2026-10-17T00:00:00Z",
    integrity_error:
      "The saved artifact summary failed its integrity check. You can delete this record from retained data.",
  };
  await page.route("**/api/crosscoders/captures", (route) =>
    route.fulfill({ json: removed ? [] : [issue] }),
  );
  await page.route("**/api/crosscoders", (route) =>
    route.fulfill({ json: [] }),
  );
  await page.route("**/api/crosscoders/captures/" + identifier, (route) => {
    expect(route.request().method()).toBe("DELETE");
    removed = true;
    return route.fulfill({ status: 204 });
  });
  await page.goto("/features/compare");
  await expect(
    page.locator(".crosscoder-workbench > .notice.error"),
  ).toContainText("Some saved artifact summaries failed integrity checks.");
  await expect(
    page.getByLabel("Reference capture", { exact: true }),
  ).toHaveCount(0);
  await page.getByText("Manage retained data", { exact: true }).click();
  await page
    .getByLabel("Saved artifact", { exact: true })
    .selectOption(identifier);
  await expect(page.locator(".crosscoder-storage .notice.error")).toContainText(
    "failed its integrity check",
  );
  await expect(
    page.getByRole("button", { name: "Export evidence", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Download tensors", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Delete saved artifact", exact: true })
    .click();
  await screenshot(page, "crosscoder-damaged-summary");
  await page
    .getByRole("button", { name: "Confirm deletion", exact: true })
    .click();
  await expect.poll(() => removed).toBe(true);
  await expect(
    page.getByText("Manage retained data", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.locator(".crosscoder-workbench > .notice.error"),
  ).toHaveCount(0);
  await screenshot(page, "crosscoder-recovered-summary");
});

test("capture limits follow the loaded context and resource plans disclose truncation", async ({
  page,
  request,
}) => {
  await load(request, "hf-internal-testing/tiny-random-gpt2", {
    context_length: 64,
  });
  await page.goto("/features/compare");
  const reference = page.locator("section.panel").filter({
    has: page.getByRole("heading", {
      name: "1 · Reference activations",
      exact: true,
    }),
  });
  await reference
    .getByText("Advanced capture settings", { exact: true })
    .click();
  const limit = reference.getByLabel("Tokens per document", { exact: true });
  await expect(limit).toHaveAttribute("max", "64");
  await expect(limit).toHaveValue("64");
  const corpus = reference.getByLabel("Training corpus", { exact: true });
  await corpus.fill(
    [
      "😀".repeat(4001),
      "the capital of France is",
      "the capital of Poland is",
      "France is the capital",
    ].join("\n"),
  );
  await expect(
    reference.getByRole("button", { name: "Capture reference", exact: true }),
  ).toBeEnabled();
  await corpus.fill(
    ["France", "Poland", "capital", "is"]
      .map((prefix) => prefix + " the".repeat(80))
      .join("\n"),
  );
  let captures = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      request.url().endsWith("/api/crosscoders/captures")
    )
      captures++;
  });
  await reference
    .getByRole("button", { name: "Check resources", exact: true })
    .click();
  await expect(reference).toContainText("Capture limit 64 tokens per document");
  await expect(reference).toContainText(
    "4 documents exceed the capture context and will be truncated",
  );
  expect(captures).toBe(0);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await reference.locator(".feature-plan").scrollIntoViewIfNeeded();
  await screenshot(page, "crosscoder-context-truncation");
});
