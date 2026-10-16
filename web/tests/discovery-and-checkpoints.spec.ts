import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

test("Hub discovery applies ranking, time, popularity and author filters", async ({
  page,
}) => {
  const seen: URL[] = [];
  await page.route("**/api/models?**", async (route) => {
    const url = new URL(route.request().url());
    seen.push(url);
    if (
      url.searchParams.get("sort") === "recommended" &&
      !url.searchParams.get("q")
    ) {
      return route.fulfill({
        json: [
          {
            id: "test/recommended",
            name: "Small model",
            family: "Llama",
            parameters: "135M",
            description:
              "A compact starting point for real circuit exploration.",
            tag: "Recommended",
            color: "orange",
          },
        ],
      });
    }
    await route.fulfill({
      json: [
        {
          id: "test/ranked",
          name: "Ranked model",
          family: "transformers",
          parameters: "135M",
          description: "test/ranked",
          tag: "Hugging Face",
          color: "blue",
          downloads: 123456,
          likes: 321,
          updated_at: "2026-10-14T10:00:00Z",
        },
      ],
    });
  });
  await page.goto("/");
  await expect(page.getByRole("button", { name: /Small model/ })).toBeVisible();
  await page
    .getByLabel("Sort models", { exact: true })
    .selectOption("trending");
  await expect(
    page.getByRole("button", { name: /Ranked model/ }),
  ).toBeVisible();
  await expect(page.getByLabel("321 likes", { exact: true })).toBeVisible();
  await page.getByText("Filters", { exact: true }).click();
  await page.getByLabel("Model time window").selectOption("30");
  await page.getByLabel("Minimum likes", { exact: true }).selectOption("100");
  await page.getByLabel("Author or organization", { exact: true }).fill("test");
  await page.getByLabel("Search Hugging Face models").fill("ranked");
  await page
    .getByRole("button", { name: "Apply filters", exact: true })
    .click();
  await expect.poll(() => seen.at(-1)?.searchParams.get("author")).toBe("test");
  expect(seen.at(-1)?.searchParams.get("sort")).toBe("trending");
  expect(seen.at(-1)?.searchParams.get("min_likes")).toBe("100");
  expect(seen.at(-1)?.searchParams.get("days")).toBe("30");
  await expect(page.getByText(/first 500 Hub results/)).toBeVisible();
  await expect(page.locator(".sidebar nav .nav-item")).toHaveCount(5);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await capturePage(page, {
    path: "../artifacts/screenshots/library-discovery.png",
    fullPage: false,
  });
});

test("block dataflow exposes attention branches and opens actual matrices", async ({
  page,
}) => {
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "Block dataflow", exact: true })
    .click();
  await expect(
    page.getByLabel("Transformer block computation flow"),
  ).toBeVisible();
  const flow = page.locator(".block-dataflow");
  await expect(flow.locator(".react-flow__node")).toHaveCount(15);
  await expect(
    flow.getByText("Attention probabilities", { exact: true }),
  ).toBeVisible();
  await expect(
    flow.getByText("Weighted value sum", { exact: true }),
  ).toBeVisible();
  await expect(flow.locator(".react-flow__edge[data-id]")).toHaveCount(18);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  async function expectNodeInside(id: string) {
    await expect
      .poll(async () => {
        const box = await flow
          .locator('.react-flow__node[data-id="' + id + '"]')
          .boundingBox();
        const area = await flow.boundingBox();
        return (
          !!box &&
          !!area &&
          box.y >= area.y &&
          box.y + box.height <= area.y + area.height &&
          box.x >= area.x &&
          box.x + box.width <= area.x + area.width
        );
      })
      .toBe(true);
  }
  await page.getByLabel("Diagram focus", { exact: true }).selectOption("mlp");
  await expectNodeInside("down");
  await page.getByLabel("Diagram focus", { exact: true }).selectOption("all");
  await expectNodeInside("input");
  await expectNodeInside("output");
  await page
    .getByLabel("Diagram focus", { exact: true })
    .selectOption("attention");
  await expectNodeInside("qkv");
  await capturePage(page, {
    path: "../artifacts/screenshots/block-dataflow-light.png",
    fullPage: true,
  });
  await flow
    .getByRole("button", {
      name: "Inspect transformer.h.0.attn.c_attn.weight",
      exact: true,
    })
    .click();
  await expect(
    page.getByText("Singular value spectrum", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Focus on transformer block").selectOption("2");
  await expect(
    flow.getByRole("button", {
      name: "Inspect transformer.h.2.attn.c_attn.weight",
      exact: true,
    }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/block-dataflow-mobile.png",
    fullPage: true,
  });
});

test("a researcher compares real base and finetuned checkpoints and reopens evidence", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  const before = await (await request.get("/api/runtime")).json();
  let reportId = "",
    referenceId = "";
  try {
    await page.goto("/workspace");
    await page
      .getByRole("button", { name: "Compare checkpoints", exact: true })
      .click();
    await page
      .getByLabel("Reference name", { exact: true })
      .fill("Base versus finetuned");
    await page
      .getByLabel("Comparison prompt", { exact: true })
      .fill("the capital of France is");
    await page
      .getByRole("button", { name: "Capture reference", exact: true })
      .click();
    await expect(
      page.getByLabel("Reference capture", { exact: true }),
    ).not.toHaveValue("");
    referenceId = await page
      .getByLabel("Reference capture", { exact: true })
      .inputValue();
    await page
      .getByRole("link", { name: "Load another checkpoint", exact: true })
      .click();
    await page
      .getByLabel("Search Hugging Face models")
      .fill("circuit-lantern-test/finetuned");
    await page
      .getByRole("button", { name: "Check model", exact: true })
      .click();
    await expect(
      page.getByText("Ready to explore on this server", { exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Download & explore", exact: true })
      .click();
    await expect(page).toHaveURL(/workspace\?view=compare/);
    await expect(
      page.getByLabel("Reference capture", { exact: true }),
    ).toHaveValue(referenceId);
    await expect(
      page.getByText("Loaded checkpoint", { exact: false }),
    ).toContainText("circuit-lantern-test/finetuned");
    await page
      .getByRole("button", { name: "Compare with loaded model", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Checkpoint changes", exact: true }),
    ).toBeVisible();
    reportId = await page
      .getByLabel("Open a comparison report", { exact: true })
      .inputValue();
    const report = await (
      await request.get("/api/checkpoints/comparisons/" + reportId)
    ).json();
    expect(report.selected.model_id).toBe("circuit-lantern-test/finetuned");
    expect(report.weights[0].distance).toBeGreaterThan(0);
    expect(
      report.layers.some((item: { distance: number }) => item.distance > 0),
    ).toBe(true);
    expect(report.kl_reference_to_selected).toBeGreaterThanOrEqual(0);
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await capturePage(page, {
      path: "../artifacts/screenshots/checkpoint-comparison-light.png",
      fullPage: true,
    });
    const download = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export comparison", exact: true })
      .click();
    const stream = await (await download).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    expect(JSON.parse(Buffer.concat(chunks).toString()).id).toBe(reportId);
    await page.reload();
    await page
      .getByLabel("Open a comparison report", { exact: true })
      .selectOption(reportId);
    await expect(
      page.getByRole("heading", { name: "Checkpoint changes", exact: true }),
    ).toBeVisible();
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await capturePage(page, {
      path: "../artifacts/screenshots/checkpoint-comparison-mobile.png",
      fullPage: true,
    });
  } finally {
    if (reportId)
      await request.delete("/api/checkpoints/comparisons/" + reportId);
    if (referenceId)
      await request.delete("/api/checkpoints/references/" + referenceId);
    const jobResponse = await request.post("/api/models/load", {
      data: {
        model_id: before.metadata.id,
        revision: before.metadata.revision,
        device: before.device,
        precision: before.metadata.execution_precision || "auto",
        context_length: before.metadata.execution_context_length || 256,
      },
    });
    expect(jobResponse.status()).toBe(202);
    const job = await jobResponse.json();
    await expect
      .poll(
        async () =>
          (await (await request.get("/api/jobs/" + job.id)).json()).status,
      )
      .toBe("ready");
  }
});
