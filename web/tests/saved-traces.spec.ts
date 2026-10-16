import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

async function capture(page: Page, prompt: string) {
  await page.getByLabel("Prompt", { exact: true }).fill(prompt);
  await page.getByLabel("Generated tokens").selectOption("0");
  const response = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace" }).click();
  const trace = await (await response).json();
  await expect(page.getByLabel("Captured prompt")).toHaveText(
    trace.decoded_prompt ?? trace.prompt,
  );
  return trace;
}

test("a delayed save stays attached to its original capture while a newer capture remains unsaved", async ({
  page,
  request,
}) => {
  const records: string[] = [];
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let stored!: () => void;
  const committed = new Promise<void>((resolve) => {
    stored = resolve;
  });
  let delivered!: () => void;
  const completed = new Promise<void>((resolve) => {
    delivered = resolve;
  });
  let firstRecord: { id: string } | undefined;
  try {
    await page.goto("/playground");
    const first = await capture(page, "the capital of France");
    await page.getByLabel("Experiment name").fill("First capture pending save");
    await page.route("**/api/experiments", async (route) => {
      if (route.request().method() !== "POST") return route.continue();
      const response = await route.fetch();
      firstRecord = await response.json();
      records.push(firstRecord!.id);
      stored();
      await gate;
      await route.fulfill({ response });
      delivered();
    });
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    await committed;
    await expect(
      page.getByRole("button", { name: "Saving experiment", exact: true }),
    ).toBeDisabled();
    const second = await capture(page, "the capital of Italy");
    await expect(page.getByLabel("Experiment name")).toHaveValue(/Italy/);
    await expect(
      page.getByRole("button", { name: "Save experiment", exact: true }),
    ).toBeEnabled();
    release();
    await completed;
    await page.unroute("**/api/experiments");
    await expect(
      page.getByRole("link", { name: "View saved experiment", exact: true }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Save experiment", exact: true }),
    ).toBeEnabled();
    await page
      .getByLabel("Experiment name")
      .fill("Second capture saved directly");
    const saving = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/experiments") &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    const secondRecord = await (await saving).json();
    records.push(secondRecord.id);
    expect(
      (await (await request.get("/api/experiments/" + secondRecord.id)).json())
        .trace,
    ).toEqual(second);
    expect(
      (await (await request.get("/api/experiments/" + firstRecord!.id)).json())
        .trace,
    ).toEqual(first);
    const link = page.getByRole("link", {
      name: "View saved experiment",
      exact: true,
    });
    await expect(link).toHaveAttribute(
      "href",
      "/experiments?experiment=" + secondRecord.id,
    );
    await capturePage(page, {
      path: "../artifacts/screenshots/trace-saved-direct.png",
    });
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await capturePage(page, {
      path: "../artifacts/screenshots/trace-saved-direct-mobile.png",
    });
    await link.click();
    await expect(page).toHaveURL(new RegExp("experiment=" + secondRecord.id));
    await expect(
      page.getByRole("heading", { name: secondRecord.name, exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel("Captured prompt")).toHaveText(
      second.decoded_prompt ?? second.prompt,
    );
  } finally {
    release();
    await page.unrouteAll({ behavior: "wait" });
    for (const id of records) await request.delete("/api/experiments/" + id);
  }
});

test("a failed recapture preserves the saved record link and a repeated capture starts a fresh save form", async ({
  page,
  request,
}) => {
  let recordId: string | undefined;
  const prompt = "the capital of France";
  try {
    await page.goto("/playground");
    const trace = await capture(page, prompt);
    const saving = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/experiments") &&
        response.request().method() === "POST",
    );
    await page
      .getByLabel("Experiment name")
      .fill("Preserved capture after failure");
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    const record = await (await saving).json();
    recordId = record.id;
    const link = page.getByRole("link", {
      name: "View saved experiment",
      exact: true,
    });
    await expect(link).toHaveAttribute(
      "href",
      "/experiments?experiment=" + recordId,
    );
    await page.route("**/api/trace", (route) =>
      route.fulfill({
        status: 409,
        json: {
          detail: "The model is busy. Try again after the current operation.",
        },
      }),
    );
    await page.getByRole("button", { name: "Run & trace" }).click();
    await expect(page.getByRole("main").getByRole("alert")).toContainText(
      "model is busy",
    );
    await expect(page.getByLabel("Experiment name")).toHaveValue(record.name);
    await expect(link).toHaveAttribute(
      "href",
      "/experiments?experiment=" + recordId,
    );
    expect(
      (await (await request.get("/api/experiments/" + recordId)).json()).trace,
    ).toEqual(trace);
    await page.unroute("**/api/trace");
    await capture(page, prompt);
    await expect(
      page.getByRole("button", { name: "Save experiment", exact: true }),
    ).toBeEnabled();
    await expect(link).toHaveCount(0);
    await expect(page.getByLabel("Experiment name")).not.toHaveValue(
      record.name,
    );
  } finally {
    await page.unrouteAll({ behavior: "wait" });
    if (recordId) await request.delete("/api/experiments/" + recordId);
  }
});
