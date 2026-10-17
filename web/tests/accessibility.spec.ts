import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

for (const [path, title] of Object.entries({
  "/": "Model library",
  "/workspace": "Model explorer",
  "/hardware": "Server & runtime",
  "/playground": "Activation lab",
  "/causal": "Residual patching",
  "/causal/heads": "Head ablation",
  "/experiments": "Experiments",
  "/guide": "Field guide",
})) {
  test(`page ${path} has accessible controls and readable contrast`, async ({
    page,
  }) => {
    await page.goto(path, { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await expect(page).toHaveTitle(title + " · Circuit Lantern");
    await expect(
      page.getByText("Backend connected", { exact: true }),
    ).toBeVisible();
    await page.waitForTimeout(300);
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    const violations = results.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => ({
        target: n.target,
        summary: n.failureSummary,
      })),
    }));
    expect(violations).toEqual([]);
  });
}
