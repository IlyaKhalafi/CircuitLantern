import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { RuntimeState } from "../src/lib/types";
import { capturePage } from "./helpers";

let previous: RuntimeState | null = null;

test.beforeEach(async ({ request }) => {
  previous = await (await request.get("/api/runtime")).json();
});

test.afterEach(async ({ request }) => {
  if (!previous?.loaded || !previous.metadata) return;
  const response = await request.post("/api/models/load", {
    data: {
      model_id: previous.metadata.id,
      revision: previous.metadata.revision,
      precision: previous.precision ?? "auto",
      device: previous.device,
      context_length: previous.context_length ?? 128,
    },
  });
  expect(response.status()).toBe(202);
  const job = await response.json();
  await expect
    .poll(
      async () =>
        (await (await request.get(`/api/jobs/${job.id}`)).json()).status,
    )
    .toBe("ready");
});

test("native bitsandbytes FP4 stays distinct from NF4 across deployed and saved views", async ({
  page,
  request,
}) => {
  const hardware = await (await request.get("/api/hardware")).json();
  test.skip(
    !hardware.quantization?.cpu?.int4?.supported,
    "A usable CPU four bit kernel is required.",
  );
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  await page.getByText("Checkpoint revision", { exact: true }).click();
  await page.getByLabel("Hub revision").fill("fp4");
  await page.getByRole("button", { name: "Check revision" }).click();
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  await expect(
    page.getByRole("option", { name: "FP4 · bitsandbytes", exact: true }),
  ).toHaveCount(1);
  await page.getByText("How this estimate works").click();
  await expect(page.getByText(/FP4 \/ FP32 residuals/)).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/compatibility-fp4.png",
  });
  await page
    .getByRole("button", { name: "Download & explore", exact: true })
    .click();
  await expect(page).toHaveURL(/workspace/);
  await page.getByLabel("Filter matrices").fill("h.0.attn.c_attn");
  await page
    .locator(".weight-list")
    .getByRole("button", {
      name: /transformer.h.0.attn.c_attn.weight/,
    })
    .click();
  await expect(page.getByText(/96 × 32 · fp4 source/)).toBeVisible();
  const runtime = await (await request.get("/api/runtime")).json();
  expect(
    runtime.metadata.execution_config.quantization.bnb_4bit_quant_type,
  ).toBe("fp4");

  await page.goto("/hardware");
  await expect(
    page.getByText("FP4 · bitsandbytes", { exact: true }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/hardware-fp4.png",
  });

  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("1");
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("FP4 · bitsandbytes", { exact: true }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/activation-fp4.png",
  });

  await page.goto("/causal/heads");
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("FP4 · bitsandbytes", { exact: true }),
  ).toBeVisible();

  await page.goto("/causal");
  await page
    .getByRole("button", { name: "Run patch sweep", exact: true })
    .click();
  await expect(page.getByText("Sweep complete", { exact: true })).toBeVisible();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("FP4 · bitsandbytes", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Experiment name").fill("Native FP4 causal evidence");
  await page
    .getByRole("button", { name: "Save experiment", exact: true })
    .click();
  await page.getByRole("link", { name: "Open notebook", exact: true }).click();
  await page.getByRole("link", { name: /Native FP4 causal evidence/ }).click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("FP4 · bitsandbytes", { exact: true }),
  ).toBeVisible();
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await capturePage(page, {
    path: "../artifacts/screenshots/notebook-fp4.png",
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/notebook-fp4-mobile.png",
  });
});
