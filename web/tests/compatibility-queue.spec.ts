import { expect, test } from "@playwright/test";

test("rapid deployment changes check only the latest selection without overlapping fits", async ({
  page,
}) => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const payloads: Record<string, unknown>[] = [];
  let firstComplete = false;
  let overlaps = 0;
  await page.route("**/api/models/compatibility", async (route) => {
    payloads.push(route.request().postDataJSON());
    if (payloads.length === 1) {
      entered();
      await held;
      const response = await route.fetch();
      firstComplete = true;
      await route.fulfill({ response });
    } else if (!firstComplete) {
      overlaps++;
      await route.fulfill({
        status: 409,
        json: {
          detail:
            "The model is busy with another operation. Wait for it to finish or cancel the active sweep.",
        },
      });
    } else {
      await route.fulfill({ response: await route.fetch() });
    }
  });
  try {
    await page.goto("/");
    await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
    await started;
    await page
      .getByLabel("Context allowance", { exact: true })
      .selectOption("64");
    await page
      .getByLabel("Execution format", { exact: true })
      .selectOption("int4");
    await page
      .getByLabel("Context allowance", { exact: true })
      .selectOption("128");
  } finally {
    release();
  }
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  expect(overlaps).toBe(0);
  expect(payloads).toHaveLength(2);
  expect(payloads[1]).toMatchObject({
    precision: "int4",
    context_length: 128,
    device: "cpu",
  });
  await page.getByText("How this estimate works").click();
  await expect(page.getByText(/NF4 \/ FP32 compute/)).toBeVisible();
});
