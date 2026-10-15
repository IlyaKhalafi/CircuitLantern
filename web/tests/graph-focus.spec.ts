import { expect, test } from "@playwright/test";

test("a researcher can focus a block and inspect a matrix directly", async ({
  page,
  request,
}) => {
  const runtime = await (await request.get("/api/runtime")).json();
  test.skip(
    !runtime.loaded,
    "A model is required to inspect its architecture.",
  );
  await page.goto("/workspace");
  await page.getByLabel("Focus on transformer block").selectOption("0");
  await expect(
    page.getByText("Block 0 matrices", { exact: true }),
  ).toBeVisible();
  const graph = page.locator(".graph-container");
  const matrix = graph
    .getByRole("button", { name: /Inspect .*?(q_proj|c_attn).*?weight/ })
    .first();
  await expect(matrix).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot({
    path: "../artifacts/screenshots/graph-block-focus.png",
    fullPage: true,
  });
  await matrix.click();
  await expect(
    page.getByText("Singular value spectrum", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Focus on transformer block").selectOption("all");
  await expect(
    page.getByText("All weight matrices", { exact: true }),
  ).toBeVisible();
});
