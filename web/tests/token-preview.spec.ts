import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

test("prompt preview exposes exact vocabulary IDs, special tokens and capture limits", async ({
  page,
}) => {
  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("4");
  await page.getByLabel("Prompt", { exact: true }).fill("<eos> the capital");
  await expect(page.getByText(/Prompt 3 \/ \d+ tokens/)).toBeVisible();
  await page.getByText("Inspect tokenization", { exact: true }).click();
  const table = page.getByRole("table", { name: "Prompt tokenization" });
  await expect(table.getByRole("row").nth(1)).toContainText("<eos>");
  await expect(table.getByRole("row").nth(1)).toContainText("Special");
  await expect(
    table.getByRole("row").nth(1).getByRole("cell", { name: "1", exact: true }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/prompt-token-preview.png",
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/prompt-token-preview-mobile.png",
  });
  await page.setViewportSize({ width: 1440, height: 1100 });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the ".repeat(140).trim());
  await expect(
    page.getByText(/Prompt exceeds the .* token capture allowance/),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run & trace" }),
  ).toBeDisabled();
  await page.getByLabel("Prompt", { exact: true }).fill("the capital");
  await expect(page.getByText(/Prompt 2 \/ \d+ tokens/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Run & trace" })).toBeEnabled();
});

test("token previews discard delayed responses for previous prompt text", async ({
  page,
}) => {
  let entered!: () => void;
  let release!: () => void;
  let finished!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const waiting = new Promise<void>((resolve) => {
    release = resolve;
  });
  const completed = new Promise<void>((resolve) => {
    finished = resolve;
  });
  await page.route("**/api/tokenize", async (route) => {
    const response = await route.fetch();
    if (route.request().postDataJSON().prompts[0] === "the capital") {
      entered();
      await waiting;
      await route.fulfill({ response }).catch(() => {});
      finished();
    } else {
      await route.fulfill({ response });
    }
  });
  await page.goto("/playground");
  await page.getByLabel("Prompt", { exact: true }).fill("the capital");
  await started;
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France");
  await expect(page.getByText(/Prompt 4 \/ \d+ tokens/)).toBeVisible();
  release();
  await completed;
  await expect(page.getByText(/Prompt 4 \/ \d+ tokens/)).toBeVisible();
  await expect(page.getByText(/Prompt 2 \/ \d+ tokens/)).toHaveCount(0);
});

test("causal token alignment prevents mismatched prompts and invalid patch positions", async ({
  page,
}) => {
  await page.goto("/causal");
  await page.getByLabel("Clean prompt").fill("the capital of France");
  await page.getByLabel("Corrupted prompt").fill("the capital");
  await expect(
    page.getByText(/Clean and corrupted prompts have different token counts/),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run patch sweep" }),
  ).toBeDisabled();
  await page.getByText("Inspect tokenization", { exact: true }).click();
  await expect(
    page
      .getByRole("table", { name: "Prompt token alignment" })
      .getByText("Missing token"),
  ).toHaveCount(2);
  await page.getByLabel("Corrupted prompt").fill("the capital of Poland");
  await expect(page.getByText(/Corrupted 4 \/ \d+ tokens/)).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run patch sweep" }),
  ).toBeEnabled();
  await expect(
    page
      .getByRole("table", { name: "Prompt token alignment" })
      .getByRole("row")
      .last(),
  ).toContainText("Changed");
  await page.getByLabel("Positions to restore").selectOption("custom");
  await page.getByLabel("Patch token position").fill("4");
  await expect(
    page.getByText("Choose a patch position within the tokenized prompts."),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run patch sweep" }),
  ).toBeDisabled();
  await page.getByLabel("Patch token position").fill("3");
  await expect(
    page.getByRole("button", { name: "Run patch sweep" }),
  ).toBeEnabled();
  await capturePage(page, {
    path: "../artifacts/screenshots/causal-token-preview.png",
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await page.getByRole("button", { name: "Run patch sweep" }).click();
  await expect(page.getByText("Sweep complete", { exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/causal-token-preview-mobile.png",
  });
});

test("an unavailable preview leaves capture usable with server validation", async ({
  page,
}) => {
  await page.route("**/api/tokenize", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Preview unavailable." }),
    }),
  );
  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("4");
  await page.getByLabel("Prompt", { exact: true }).fill("the capital");
  await expect(
    page.getByText(/Token preview is temporarily unavailable/),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Run & trace" })).toBeEnabled();
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
});
