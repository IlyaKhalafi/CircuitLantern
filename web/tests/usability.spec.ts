import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";
import fa from "../src/i18n/messages/fa.json";
import ar from "../src/i18n/messages/ar.json";

test("page names match navigation and primary library controls appear first", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 1050 });
  for (const path of [
    "/",
    "/workspace",
    "/playground",
    "/features",
    "/causal",
    "/causal/heads",
    "/causal/heads/induction",
    "/experiments",
    "/hardware",
    "/guide",
  ]) {
    await page.goto(path);
    await expect(page.locator("h1")).toHaveText(
      await page.locator(".breadcrumb strong").innerText(),
    );
    await expect(
      page.locator(".page-header .eyebrow, .sidebar-note, .workspace-label"),
    ).toHaveCount(0);
  }
  await page.goto("/");
  await expect(page.locator(".model-card").first()).toBeVisible();
  const search = (await page
    .getByLabel("Search Hugging Face models")
    .boundingBox())!;
  expect(search.y + search.height).toBeLessThan(400);
  for (const card of (await page.locator(".model-card").all()).slice(0, 3)) {
    const box = (await card.boundingBox())!;
    expect(box.y + box.height).toBeLessThan(1050);
  }
  await capturePage(page, {
    path: "../artifacts/screenshots/usability-library.png",
    fullPage: false,
  });
});

test("help works with keyboard, Escape and hovering over its content", async ({
  page,
}) => {
  await page.goto("/workspace");
  const trigger = page.getByRole("button", {
    name: "About Parameters",
    exact: true,
  });
  await expect(trigger).toHaveAccessibleDescription(
    "Measured from loaded weights",
  );
  await trigger.focus();
  await page.keyboard.press("Enter");
  const tooltip = page.getByRole("tooltip");
  await expect(tooltip).toHaveText("Measured from loaded weights");
  await expect(trigger).toHaveAccessibleDescription(
    "Measured from loaded weights",
  );
  await page.keyboard.press("Escape");
  await expect(tooltip).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await page
    .getByRole("heading", { name: "Model explorer", exact: true })
    .click();
  await trigger.hover();
  await expect(tooltip).toBeVisible();
  await tooltip.hover();
  await expect(tooltip).toBeVisible();
  await page.mouse.move(0, 0);
  await expect(tooltip).toHaveCount(0);
});

test("only the current help stays open and navigation dismisses it", async ({
  page,
}) => {
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "About Parameters", exact: true })
    .click();
  await expect(page.getByRole("tooltip")).toHaveText(
    "Measured from loaded weights",
  );
  await page
    .getByRole("button", { name: "About Weight matrices", exact: true })
    .click();
  await expect(page.getByRole("tooltip")).toHaveCount(1);
  await expect(page.getByRole("tooltip")).toHaveText(
    "Two dimensional parameters",
  );
  await page.locator('.sidebar nav a[href="/playground"]').click();
  await expect(page.getByRole("tooltip")).toHaveCount(0);
});

test("modal help stays above the dialog and Escape closes help first", async ({
  page,
}) => {
  let fits = 0;
  await page.route("**/api/models/compatibility", async (route) => {
    fits++;
    await route.continue();
  });
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  const trigger = page.getByRole("button", {
    name: "About Execution format",
    exact: true,
  });
  await trigger.click();
  const tooltip = page.getByRole("tooltip");
  await expect(tooltip).toBeVisible();
  expect(
    await tooltip.evaluate((node) => Boolean(node.closest("dialog"))),
  ).toBe(true);
  expect(
    await tooltip.evaluate((node) => {
      const box = node.getBoundingClientRect();
      return !!document
        .elementFromPoint(box.x + box.width / 2, box.y + box.height / 2)
        ?.closest('[role="tooltip"]');
    }),
  ).toBe(true);
  await page.keyboard.press("Escape");
  await expect(tooltip).toHaveCount(0);
  await expect(page.locator("dialog")).toBeVisible();
  await expect(trigger).toBeFocused();
  expect(fits).toBe(1);
  await page.keyboard.press("Escape");
  await expect(page.locator("dialog")).toHaveCount(0);
});

for (const locale of ["fa", "ar"] as const)
  for (const theme of ["light", "dark"]) {
    test(`touch help stays visible and dismissible in ${locale} ${theme}`, async ({
      browser,
      baseURL,
    }) => {
      const context = await browser.newContext({
        baseURL,
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      });
      await context.addCookies([
        { name: "lantern_locale", value: locale, url: baseURL! },
        { name: "lantern_theme", value: theme, url: baseURL! },
      ]);
      const page = await context.newPage();
      try {
        await page.goto("/workspace");
        const messages = locale === "fa" ? fa : ar;
        const trigger = page.getByRole("button", {
          name: messages["About {label}"].replace(
            "{label}",
            messages.Parameters,
          ),
          exact: true,
        });
        await trigger.tap();
        const tooltip = page.getByRole("tooltip");
        await expect(tooltip).toBeVisible();
        const box = (await tooltip.boundingBox())!;
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.x + box.width).toBeLessThanOrEqual(390);
        // Capture help without the general helper, which blurs focus and scrolls.
        await page.screenshot({
          path: `../artifacts/screenshots/usability-help-${locale}-${theme}.png`,
          fullPage: false,
        });
        await expect(tooltip).toBeVisible();
        await trigger.tap();
        await expect(tooltip).toHaveCount(0);
        await trigger.tap();
        await expect(tooltip).toBeVisible();
        await page.locator("h1").tap();
        await expect(tooltip).toHaveCount(0);
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBe(true);
      } finally {
        await context.close();
      }
    });
  }

test("execution settings and unmet training requirements remain visible without help", async ({
  page,
}) => {
  for (const path of ["/playground", "/causal", "/causal/heads", "/features"]) {
    await page.goto(path);
    const execution = page.locator(".page-header .execution-context");
    await expect(execution).toContainText("FP32");
    await expect(execution).toContainText("CPU");
    await expect(execution).toContainText("context tokens");
    await expect(execution).toHaveAccessibleName(/FP32.*CPU/);
  }
  await page.goto("/features");
  const train = page.getByRole("button", {
    name: "Train dictionary",
    exact: true,
  });
  await expect(train).toBeDisabled();
  let trainingRequests = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/api/features/train")) trainingRequests++;
  });
  await train.focus();
  await expect(train).toBeFocused();
  await expect(train).toHaveAccessibleDescription(
    "Check training resources before training.",
  );
  await train.press("Enter");
  await train.click({ force: true });
  expect(trainingRequests).toBe(0);
  const requirement = await train.getAttribute("aria-describedby");
  await expect(page.locator(`[id="${requirement}"]`)).toHaveText(
    "Check training resources before training.",
  );
  await expect(
    page.getByText("One document per line.", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText(
      "Reconstruction and sparsity do not prove that a feature has a single semantic meaning. Inspect examples and test interventions.",
      { exact: true },
    ),
  ).toBeVisible();
});

test("unsupported head methods offer a clear route back to the model library", async ({
  page,
}) => {
  await page.route("**/api/runtime", async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    await route.fulfill({ json: { ...data, attention_head_targets: [] } });
  });
  await page.route("**/api/induction/capabilities", async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    await route.fulfill({
      json: {
        ...data,
        supported: false,
        reason:
          "This architecture has no complete verified attention head layout.",
      },
    });
  });
  await page.goto("/causal/heads");
  await expect(page.locator(".method-unavailable")).toHaveAttribute(
    "role",
    "status",
  );
  await page
    .getByRole("link", { name: "Induction probe", exact: true })
    .click();
  await expect(page).toHaveURL(/\/causal\/heads\/induction$/);
  await expect(page.locator(".method-unavailable")).toContainText(
    "no complete verified attention head layout",
  );
  await page.unrouteAll({ behavior: "wait" });
  await page
    .getByRole("link", { name: "Open a supported model", exact: true })
    .click();
  await expect(page).toHaveURL(/\/$/);
});

test("tooltip descriptions keep the explorer accessible", async ({ page }) => {
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "About Parameters", exact: true })
    .click();
  await expect(page.getByRole("tooltip")).toBeVisible();
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
});
