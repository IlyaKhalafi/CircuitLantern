import { expect, test } from "@playwright/test";

test("activation patching runs on the loaded checkpoint and exports evidence", async ({
  page,
  request,
}) => {
  const state = await request.get("/api/runtime");
  const runtime = await state.json();
  test.skip(
    !runtime.loaded,
    "A loaded causal model is required for this live workflow.",
  );

  await page.goto("/causal", { waitUntil: "domcontentloaded" });
  await expect(page.getByLabel("Clean prompt")).toBeVisible();
  await page.getByRole("button", { name: "Run patch sweep" }).click();
  await expect(page.getByText("Sweep complete", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Per block patching results", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Token alignment", { exact: true }),
  ).toBeVisible();
  await page.getByRole("main").focus();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: "../artifacts/screenshots/causal-desktop.png",
    fullPage: true,
  });

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export sweep" }).click();
  expect((await download).suggestedFilename()).toBe("causal-patching.json");

  await page.goto("/guide");
  await page.goto("/causal");
  await expect(page.getByText("Sweep complete", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Clean prompt")).toHaveValue(
    "The capital of France is",
  );

  await page.getByLabel("Positions to restore").selectOption("all");
  await page.getByRole("button", { name: "Run patch sweep" }).click();
  await expect(
    page.getByText("All token positions restored", { exact: true }),
  ).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    ),
  ).toBe(false);
  await page.getByRole("main").focus();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: "../artifacts/screenshots/causal-mobile.png",
    fullPage: true,
  });

  await page.getByLabel("Corrupted prompt").fill("The capital");
  await page.getByRole("button", { name: "Run patch sweep" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "same length",
  );
  await expect(
    page.getByRole("button", { name: "Run patch sweep" }),
  ).toBeEnabled();
});

test("stale background jobs recover after a backend restart", async ({
  page,
}) => {
  await page.addInitScript(() =>
    sessionStorage.setItem("lantern-causal-job", "missing-job"),
  );
  await page.goto("/causal", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "not found",
  );
  if (await page.getByLabel("Clean prompt").isVisible()) {
    await expect(
      page.getByRole("button", { name: "Run patch sweep" }),
    ).toBeEnabled();
  }
  expect(
    await page.evaluate(() => sessionStorage.getItem("lantern-causal-job")),
  ).toBeNull();
});
