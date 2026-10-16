import { capturePage } from "./helpers";
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

async function expectAccessible(page: import("@playwright/test").Page) {
  const result = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    result.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => ({
        target: n.target,
        summary: n.failureSummary,
      })),
    })),
  ).toEqual([]);
}

test("deployment dialog supports keyboard focus and readable controls", async ({
  page,
}) => {
  await page.goto("/");
  const card = page.getByRole("button", { name: /Tiny random GPT-2/ });
  await card.click();
  const dialog = page.getByRole("dialog");
  await expect(
    dialog.getByText("Ready to explore on this server"),
  ).toBeVisible();
  await expectAccessible(page);
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(card).toBeFocused();
});

test("weight inspector charts and matrix labels remain accessible", async ({
  page,
  request,
}) => {
  const runtime = await (await request.get("/api/runtime")).json();
  test.skip(!runtime.loaded, "A model must be loaded for weight inspection.");
  await page.goto("/workspace");
  await page
    .getByLabel("Filter matrices")
    .fill(
      runtime.metadata.architecture === "gpt2"
        ? "h.0.attn.c_attn"
        : "layers.0.self_attn.q_proj",
    );
  await page.locator(".weight-item").first().click();
  await expect(
    page.getByText("Singular value spectrum", { exact: true }),
  ).toBeVisible();
  await expectAccessible(page);
  await page.getByRole("main").focus();
  await page.evaluate(() => window.scrollTo(0, 0));
  await capturePage(page, {
    path: "../artifacts/screenshots/matrix-accessible.png",
    fullPage: true,
  });
});

test("activation results expose readable charts and attention controls", async ({
  page,
  request,
}) => {
  const runtime = await (await request.get("/api/runtime")).json();
  test.skip(!runtime.loaded, "A model must be loaded for activation capture.");
  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("8");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await expectAccessible(page);
  await page.getByRole("main").focus();
  await page.evaluate(() => window.scrollTo(0, 0));
  await capturePage(page, {
    path: "../artifacts/screenshots/activation-accessible.png",
    fullPage: true,
  });
  await page.getByRole("tab", { name: "Attention", exact: true }).click();
  await expectAccessible(page);
  await page.getByRole("tab", { name: "Logit lens", exact: true }).click();
  await expectAccessible(page);
});
