import { capturePage } from "./helpers";
import { expect, test } from "@playwright/test";

test.describe.serial("research workflow", () => {
  test("model library checks hardware and loads a real checkpoint", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Open a model. Look closer." }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Tiny random GPT-2/ }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/library-desktop.png",
      fullPage: true,
    });
    await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
    await expect(
      page.getByText("Ready to explore on this server"),
    ).toBeVisible();
    await page.getByLabel("Execution format").selectOption("nvfp4");
    await expect(
      page.getByText(/Native NVFP4 requires a Blackwell/),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Download & explore" }),
    ).toBeDisabled();
    await capturePage(page, {
      path: "../artifacts/screenshots/compatibility-blocked.png",
      fullPage: true,
    });
    await page.getByLabel("Execution format").selectOption("auto");
    await expect(
      page.getByText("Ready to explore on this server"),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/compatibility-ready.png",
      fullPage: true,
    });
    await page.getByRole("button", { name: "Download & explore" }).click();
    await expect(page).toHaveURL(/workspace/);
    await expect(
      page.getByText("Transformer blocks", { exact: true }),
    ).toBeVisible();
  });

  test("architecture and weight inspector show real statistics", async ({
    page,
  }) => {
    await page.goto("/workspace");
    await expect(
      page.getByText("All weight matrices", { exact: true }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/explorer-desktop.png",
      fullPage: true,
    });
    await page.getByLabel("Filter matrices").fill("h.0.attn.c_attn");
    await page
      .locator(".weight-list")
      .getByRole("button", { name: /transformer.h.0.attn.c_attn.weight/ })
      .click();
    await expect(
      page.getByText("Singular value spectrum", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByText("Frobenius norm", { exact: true }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/matrix-inspector.png",
      fullPage: true,
    });
    await page.getByRole("tab", { name: "Weight inventory" }).click();
    await expect(
      page.getByRole("columnheader", { name: "SHAPE" }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/weight-inventory.png",
      fullPage: true,
    });
  });

  test("activation lab captures, intervenes and saves", async ({ page }) => {
    await page.goto("/playground");
    await page.getByLabel("Generated tokens").selectOption("8");
    await page.getByRole("button", { name: "Run & trace" }).click();
    await expect(
      page.getByText("Capture complete", { exact: true }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/activation-residual.png",
      fullPage: true,
    });
    await page.getByRole("tab", { name: "Logit lens" }).click();
    await expect(
      page.getByText("Watch predictions take shape", { exact: true }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/activation-logit-lens.png",
      fullPage: true,
    });
    await page.getByRole("tab", { name: "Attention", exact: true }).click();
    await expect(
      page.getByRole("img", { name: /Hover over a cell/ }),
    ).toBeVisible();
    await page.getByLabel("Attention layer").selectOption("2");
    await capturePage(page, {
      path: "../artifacts/screenshots/activation-attention.png",
      fullPage: true,
    });
    await page.getByRole("tab", { name: "Next token logits" }).click();
    await expect(
      page.getByText("Next token distribution", { exact: true }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/activation-logits.png",
      fullPage: true,
    });
    await page.getByLabel("Layer intervention").selectOption("1");
    await page.getByRole("button", { name: "Run & trace" }).click();
    await expect(
      page.getByText("Block 1 bypassed", { exact: true }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/activation-intervention.png",
      fullPage: true,
    });
    await page
      .getByLabel("Experiment name")
      .fill("Browser verification experiment");
    await page.getByRole("button", { name: "Save experiment" }).click();
    await expect(
      page.getByRole("button", { name: "Saved to experiments" }),
    ).toBeVisible();
  });

  test("experiment notebook persists notes and exports", async ({ page }) => {
    await page.goto("/experiments");
    await expect(
      page
        .getByRole("link", { name: /Browser verification experiment/ })
        .first(),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/experiments-list.png",
      fullPage: true,
    });
    await page
      .getByRole("link", { name: /Browser verification experiment/ })
      .first()
      .click();
    await page
      .getByLabel("Observations")
      .fill(
        "The bypass changes the residual trajectory. Review final normalization separately.",
      );
    await page.getByRole("button", { name: "Save notes" }).click();
    await expect(
      page.getByRole("button", { name: "Changes saved" }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/experiment-detail.png",
      fullPage: true,
    });
    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export experiment" }).click();
    expect((await download).suggestedFilename()).toMatch(/experiment-.*\.json/);
  });

  test("server and field guide pages render", async ({ page }) => {
    await page.goto("/hardware");
    await expect(
      page.getByText("Python runtime", { exact: true }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/hardware-desktop.png",
      fullPage: true,
    });
    await page.getByRole("button", { name: "Refresh snapshot" }).click();
    await expect(
      page.getByText("Physical CPU cores", { exact: true }),
    ).toBeVisible();
    await page.goto("/guide");
    await expect(
      page.getByRole("heading", { name: "Read the measurement labels" }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/guide-desktop.png",
      fullPage: true,
    });
    await page.goto("/unknown-page");
    await expect(
      page.getByText("This path is still unexplored.", { exact: true }),
    ).toBeVisible();
    await capturePage(page, {
      path: "../artifacts/screenshots/not-found.png",
      fullPage: true,
    });
  });

  test("all pages work on a phone without horizontal overflow", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    for (const path of [
      "/",
      "/workspace",
      "/playground",
      "/causal",
      "/experiments",
      "/hardware",
      "/guide",
    ]) {
      await page.goto(path);
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
      await expect(page.getByText("Backend connected")).toBeVisible();
      await page.waitForTimeout(300);
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth > window.innerWidth,
      );
      expect(overflow, `Page ${path} should fit a phone`).toBe(false);
      await capturePage(page, {
        path: `../artifacts/screenshots/${path === "/" ? "library" : path.slice(1)}-mobile.png`,
        fullPage: true,
      });
    }
    await page.getByRole("button", { name: "Open navigation" }).click();
    await page.getByRole("link", { name: "Research lab", exact: true }).click();
    await expect(page).toHaveURL(/playground/);
  });
});
