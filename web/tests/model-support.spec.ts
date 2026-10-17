import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

test("complete architecture diagnostics remain readable inside deployment checks", async ({
  page,
}) => {
  const message =
    "This architecture cannot be initialized by the installed Transformers runtime.\n" +
    "The checkpoint requires " +
    "a_new_model_type_".repeat(18) +
    ". Choose a supported checkpoint. Complete diagnostic ends here.";
  await page.route("**/api/models/compatibility", async (route) => {
    const response = await route.fetch();
    const report = await response.json();
    await route.fulfill({
      json: {
        ...report,
        compatible: false,
        status: "blocked",
        blockers: [message],
      },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  const diagnostic = page.locator(".check-message.blocked span");
  await expect(diagnostic).toHaveText(message);
  await diagnostic.scrollIntoViewIfNeeded();
  await capturePage(page, {
    path: "../artifacts/screenshots/model-support-diagnostic-light.png",
    fullPage: false,
  });
  await page.getByRole("button", { name: "Close compatibility check" }).click();
  await page.getByRole("button", { name: "Dark theme" }).click();
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await expect(diagnostic).toHaveText(message);
  await diagnostic.scrollIntoViewIfNeeded();
  await capturePage(page, {
    path: "../artifacts/screenshots/model-support-diagnostic-dark.png",
    fullPage: false,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(diagnostic).toHaveText(message);
  await expect
    .poll(() =>
      page
        .getByRole("dialog")
        .evaluate((element) => element.scrollWidth <= element.clientWidth),
    )
    .toBe(true);
  await diagnostic.scrollIntoViewIfNeeded();
  await capturePage(page, {
    path: "../artifacts/screenshots/model-support-diagnostic-mobile.png",
    fullPage: false,
  });
});

test("a Hub availability failure is shown without changing it to an access warning", async ({
  page,
}) => {
  const message =
    "Hugging Face is temporarily unable to provide this model. Try again shortly.";
  await page.route("**/api/models/compatibility", (route) =>
    route.fulfill({ status: 503, json: { detail: message } }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  const alert = page.getByRole("dialog").getByRole("alert");
  await expect(alert).toHaveText(message);
  await expect(alert).not.toContainText("HF_TOKEN");
});
