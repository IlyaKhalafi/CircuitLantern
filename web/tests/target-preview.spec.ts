import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

async function prepare(
  page: import("@playwright/test").Page,
  targets: string[],
) {
  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("0");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France is");
  await page.getByText("Track target tokens", { exact: true }).click();
  for (const [index, value] of targets.entries()) {
    await page
      .getByRole("button", { name: "Add target token", exact: true })
      .click();
    await page
      .getByLabel("Target token " + (index + 1), { exact: true })
      .fill(value);
  }
}

test("target previews expose exact IDs, pieces and duplicates without model inference", async ({
  page,
}) => {
  let forwards = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/trace"))
      forwards++;
  });
  await prepare(page, [" France", "<pad>"]);
  const fields = page.locator(".tracked-token-row");
  const run = page.getByRole("button", { name: "Run & trace", exact: true });
  await expect(fields.nth(0).locator(".target-resolution")).toContainText(
    "Vocabulary ID 6",
  );
  await expect(fields.nth(1).locator(".target-resolution")).toContainText(
    "Vocabulary ID 0",
  );
  await expect(fields.nth(1).locator(".target-resolution")).toContainText(
    "special token",
  );
  await expect(
    page.getByText("2 target tokens ready.", { exact: true }),
  ).toBeVisible();
  await fields.nth(0).getByText("Native token pieces", { exact: true }).click();
  await expect(fields.nth(0).locator("code")).toHaveText('6 · "France"');
  await expect(run).toBeEnabled();
  await capturePage(page, {
    path: "../artifacts/screenshots/target-token-preview.png",
  });
  await page.getByLabel("Target token 1", { exact: true }).fill("the capital");
  await expect(fields.nth(0)).toContainText("Split into 2 vocabulary tokens.");
  await expect(
    page.getByLabel("Target token 1", { exact: true }),
  ).toHaveAttribute("aria-invalid", "true");
  await expect(run).toBeDisabled();
  await capturePage(page, {
    path: "../artifacts/screenshots/target-token-preview-invalid.png",
  });
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
    path: "../artifacts/screenshots/target-token-preview-invalid-mobile.png",
  });
  await page.getByLabel("Target token 1", { exact: true }).fill("France");
  await page
    .getByRole("button", { name: "Add target token", exact: true })
    .click();
  await page.getByLabel("Target token 3", { exact: true }).fill(" France");
  await expect(fields.nth(2)).toContainText("already used by target 1");
  await expect(run).toBeDisabled();
  await page
    .getByRole("button", { name: "Remove target token 1", exact: true })
    .click();
  await expect(
    page.getByText("2 target tokens ready.", { exact: true }),
  ).toBeVisible();
  await expect(fields.nth(0)).toContainText("Vocabulary ID 0");
  await expect(fields.nth(1)).toContainText("Vocabulary ID 6");
  await expect(run).toBeEnabled();
  await page.setViewportSize({ width: 1440, height: 1100 });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  expect(forwards).toBe(0);
});

test("late target resolution cannot overwrite a more recent draft", async ({
  page,
}) => {
  let entered!: () => void, release!: () => void, finished!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const waiting = new Promise<void>((resolve) => {
    release = resolve;
  });
  const completed = new Promise<void>((resolve) => {
    finished = resolve;
  });
  await page.route("**/api/targets/preview", async (route) => {
    const response = await route.fetch();
    if (route.request().postDataJSON().tracked_tokens[0] === "France") {
      entered();
      await waiting;
      await route.fulfill({ response }).catch(() => {});
      finished();
    } else await route.fulfill({ response });
  });
  try {
    await prepare(page, ["France"]);
    await started;
    await page.getByLabel("Target token 1", { exact: true }).fill("Poland");
    await expect(page.locator(".target-resolution")).toContainText(
      "Vocabulary ID 9",
    );
    release();
    await completed;
    await expect(page.locator(".target-resolution")).toContainText(
      "Vocabulary ID 9",
    );
    await expect(page.locator(".target-resolution")).not.toContainText(
      "Vocabulary ID 6",
    );
    await expect(
      page.getByRole("button", { name: "Run & trace", exact: true }),
    ).toBeEnabled();
  } finally {
    release();
    await page.unrouteAll({ behavior: "wait" });
  }
});

test("unavailable target previews allow server validation and recover through retry", async ({
  page,
  request,
}) => {
  let unavailable = true;
  const runtime = await (await request.get("/api/runtime")).json();
  const sessions: string[] = [];
  await page.route("**/api/targets/preview", (route) => {
    sessions.push(route.request().postDataJSON().expected_execution_id);
    return unavailable
      ? route.fulfill({
          status: 503,
          json: { detail: "Preview temporarily unavailable" },
        })
      : route.continue();
  });
  await prepare(page, ["France", "Poland"]);
  await expect(
    page.getByText(/Target preview is temporarily unavailable/),
  ).toBeVisible();
  const run = page.getByRole("button", { name: "Run & trace", exact: true });
  await expect(run).toBeEnabled();
  const capturing = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await run.click();
  const trace = await (await capturing).json();
  expect(
    trace.tracked_targets.map(
      (target: { token_id: number }) => target.token_id,
    ),
  ).toEqual([6, 9]);
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Retry target preview", exact: true }),
  ).toBeEnabled();
  unavailable = false;
  await page
    .getByRole("button", { name: "Retry target preview", exact: true })
    .click();
  await expect(
    page.getByText("2 target tokens ready.", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText(/Target preview is temporarily unavailable/),
  ).toHaveCount(0);
  expect(sessions.length).toBeGreaterThan(1);
  expect(
    sessions.every((session) => session === runtime.metadata.execution_id),
  ).toBe(true);
});

test("long target splits retain complete counts while limiting displayed pieces", async ({
  page,
}) => {
  await prepare(page, ["the ".repeat(25)]);
  await expect(page.locator(".target-resolution")).toContainText(
    "Split into 25 vocabulary tokens.",
  );
  await page.getByText("Native token pieces", { exact: true }).click();
  await expect(page.locator(".target-resolution code")).toHaveCount(16);
  await expect(page.locator(".target-resolution")).toContainText(
    "9 more tokens",
  );
  await expect(
    page.getByRole("button", { name: "Run & trace", exact: true }),
  ).toBeDisabled();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("prompt and target previews wait their turn without overlapping tokenizer operations", async ({
  page,
}) => {
  let entered!: () => void, release!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const waiting = new Promise<void>((resolve) => {
    release = resolve;
  });
  let holding = false,
    overlapped = false,
    targetRequests = 0;
  await page.route("**/api/tokenize", async (route) => {
    const response = await route.fetch();
    holding = true;
    entered();
    await waiting;
    holding = false;
    await route.fulfill({ response }).catch(() => {});
  });
  await page.route("**/api/targets/preview", async (route) => {
    targetRequests++;
    overlapped ||= holding;
    await route.continue();
  });
  try {
    await prepare(page, ["France"]);
    await started;
    await expect(
      page.getByText("Checking target tokens", { exact: true }),
    ).toBeVisible();
    // Hold the prompt response beyond the target debounce to expose concurrent requests.
    await page.waitForTimeout(500);
    expect(targetRequests).toBe(0);
    release();
    await expect(
      page.getByText("1 target token ready.", { exact: true }),
    ).toBeVisible();
    expect(targetRequests).toBe(1);
    expect(overlapped).toBe(false);
    await expect(
      page.getByText(/preview is temporarily unavailable/),
    ).toHaveCount(0);
  } finally {
    release();
    await page.unrouteAll({ behavior: "wait" });
  }
});
