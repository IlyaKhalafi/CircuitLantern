import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

test("selected context allowance survives loading and bounds capture with generation", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await page.getByLabel("Context allowance").selectOption("16");
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  await page.getByRole("button", { name: "Download & explore" }).click();
  await expect(page).toHaveURL(/workspace/);
  const runtime = await (await request.get("/api/runtime")).json();
  expect(runtime.context_length).toBe(16);
  expect(runtime.metadata.execution_config.context_length).toBe(16);
  await page.goto("/playground");
  await expect(
    page.getByText(/Up to 8 prompt tokens with this generation length/),
  ).toBeVisible();
  await expect(page.getByLabel("Generated tokens")).toHaveValue("8");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the ".repeat(9).trim());
  await expect(
    page.getByText("Prompt 9 / 8 tokens", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run & trace" }),
  ).toBeDisabled();
  const rejected = await request.post("/api/trace", {
    data: { prompt: "the ".repeat(9).trim(), max_new_tokens: 8 },
  });
  expect(rejected.status()).toBe(400);
  expect((await rejected.json()).detail).toMatch(
    /Shorten it to at most 8 tokens/,
  );
  await page.getByLabel("Generated tokens").selectOption("4");
  await expect(
    page.getByText(/Up to 12 prompt tokens with this generation length/),
  ).toBeVisible();
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/context-bound-capture.png",
  });
  await page.getByLabel("Generated tokens").selectOption("0");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the ".repeat(16).trim());
  await expect(
    page.getByText("Prompt 16 / 16 tokens", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Run & trace" })).toBeEnabled();
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(page.getByLabel("Captured prompt")).toBeVisible();
  await expect(
    page.getByText("Prompt capture only.", { exact: false }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/context-capture-only.png",
  });
});

test("load validation preserves a model when the requested context exceeds its window", async ({
  request,
}) => {
  const before = await (await request.get("/api/runtime")).json();
  const response = await request.post("/api/models/load", {
    data: {
      model_id: "hf-internal-testing/tiny-random-gpt2",
      context_length: 1024,
    },
  });
  expect(response.status()).toBe(409);
  expect((await response.json()).detail).toMatch(/context window/);
  const after = await (await request.get("/api/runtime")).json();
  expect(after).toEqual(before);
});
