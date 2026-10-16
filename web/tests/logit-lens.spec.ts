import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Trace } from "../src/lib/types";
import { number } from "../src/lib/api";
import { capturePage } from "./helpers";

async function capture(page: Page): Promise<Trace> {
  await page.goto("/playground");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France");
  await page.getByLabel("Generated tokens").selectOption("0");
  const capturing = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  const trace = await (await capturing).json();
  await page.getByRole("tab", { name: "Logit lens", exact: true }).click();
  return trace;
}

async function expectPosition(page: Page, trace: Trace, position: number) {
  await page
    .getByRole("combobox", { name: "Logit lens token position", exact: true })
    .selectOption(String(position));
  await expect(page.locator(".position-lens-table tbody tr")).toHaveCount(
    trace.layers.length,
  );
  for (const [index, layer] of trace.layers.entries()) {
    const cells = page
      .locator(".position-lens-table tbody tr")
      .nth(index)
      .getByRole("cell");
    await expect(cells.nth(0)).toHaveText(
      layer.token_predictions[position] || "∅",
    );
    await expect(cells.nth(1)).toHaveText(
      String(layer.token_prediction_ids![position]),
    );
    await expect(cells.nth(2)).toHaveText(
      number(layer.token_prediction_probabilities![position] * 100, 4) + "%",
    );
    await expect(cells.nth(3)).toHaveText(
      number(layer.token_entropies![position], 5),
    );
    await expect(cells.nth(4)).toHaveText(
      number(layer.token_norms[position], 5),
    );
  }
}

test("logit lens confidence and entropy follow every selected prompt position without another forward pass", async ({
  page,
}) => {
  let forwards = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/api/trace") && request.method() === "POST")
      forwards++;
  });
  const trace = await capture(page);
  expect(
    trace.layers.some(
      (layer) =>
        layer.token_prediction_probabilities![0] !==
        layer.token_prediction_probabilities!.at(-1),
    ),
  ).toBe(true);
  await expectPosition(page, trace, 0);
  await expect(
    page.getByRole("status").filter({ hasText: "Prompt position 0" }),
  ).toContainText("causal prefix");
  await expectPosition(page, trace, 1);
  await page.getByRole("tab", { name: "Residual stream", exact: true }).click();
  await page.getByRole("tab", { name: "Logit lens", exact: true }).click();
  await expect(
    page.getByRole("combobox", {
      name: "Logit lens token position",
      exact: true,
    }),
  ).toHaveValue("1");
  await expectPosition(page, trace, trace.tokens.length - 1);
  await expect(
    page.getByRole("status").filter({ hasText: "Prompt position" }),
  ).toContainText("next token after the prompt");
  await page
    .getByText("Inspect final token transitions", { exact: true })
    .click();
  const transitions = page.locator(".lens-transitions-table tbody tr");
  await expect(transitions).toHaveCount(trace.layers.length);
  for (const [index, layer] of trace.layers.entries()) {
    await expect(transitions.nth(index).getByRole("cell").nth(0)).toHaveText(
      number(layer.cosine_similarity, 5),
    );
    await expect(transitions.nth(index).getByRole("cell").nth(1)).toHaveText(
      number(layer.update_norm, 5),
    );
  }
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await capturePage(page, {
    path: "../artifacts/screenshots/position-logit-lens.png",
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expectPosition(page, trace, 0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/position-logit-lens-mobile.png",
  });
  expect(forwards).toBe(1);
});

test("saved logit lens position measurements reopen and export with their original vocabulary IDs", async ({
  page,
  request,
}) => {
  let id: string | undefined;
  try {
    const trace = await capture(page);
    await page
      .getByLabel("Experiment name")
      .fill("Per-position logit lens evidence");
    const saving = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/experiments") &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    id = (await (await saving).json()).id;
    await page
      .getByRole("link", { name: "View saved experiment", exact: true })
      .click();
    await page.getByRole("tab", { name: "Logit lens", exact: true }).click();
    await expectPosition(page, trace, 0);
    await page.reload();
    await page.getByRole("tab", { name: "Logit lens", exact: true }).click();
    await expectPosition(page, trace, 1);
    const downloading = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export experiment", exact: true })
      .click();
    const stream = await (await downloading).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    const exported = JSON.parse(Buffer.concat(chunks).toString());
    expect(exported.trace).toEqual(trace);
    expect(
      (await (await request.get("/api/experiments/" + id)).json()).trace,
    ).toEqual(trace);
  } finally {
    if (id) await request.delete("/api/experiments/" + id);
  }
});

test("older captures keep final-token measurements separate from the selected position prediction", async ({
  page,
}) => {
  await page.route("**/api/trace", async (route) => {
    const response = await route.fetch();
    const trace = await response.json();
    delete trace.token_prediction_method;
    for (const layer of trace.layers) {
      delete layer.token_prediction_ids;
      delete layer.token_prediction_probabilities;
      delete layer.token_entropies;
    }
    await route.fulfill({ response, json: trace });
  });
  const trace = await capture(page);
  await expect(
    page.getByText(
      /This older capture stores confidence and entropy for the final prompt token/,
    ),
  ).toBeVisible();
  await page
    .getByRole("combobox", { name: "Logit lens token position", exact: true })
    .selectOption("0");
  await expect(
    page.getByRole("columnheader", {
      name: "LAST TOKEN PROBABILITY",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator(".position-lens-table")).toHaveCount(0);
  for (const [index, layer] of trace.layers.entries()) {
    const cells = page
      .locator(".legacy-lens-table tbody tr")
      .nth(index)
      .getByRole("cell");
    await expect(cells.nth(0)).toHaveText(layer.token_predictions[0] || "∅");
    await expect(cells.nth(1)).toHaveText(layer.predictions[0].token);
    await expect(cells.nth(2)).toHaveText(
      number(layer.predictions[0].probability * 100, 4) + "%",
    );
    await expect(cells.nth(3)).toHaveText(number(layer.entropy));
  }
  await capturePage(page, {
    path: "../artifacts/screenshots/legacy-logit-lens.png",
  });
});

test("missing position probabilities stay unavailable rather than falling back to the final token", async ({
  page,
}) => {
  await page.route("**/api/trace", async (route) => {
    const response = await route.fetch();
    const trace = await response.json();
    for (const layer of trace.layers)
      layer.token_prediction_probabilities[0] = null;
    await route.fulfill({ response, json: trace });
  });
  const trace = await capture(page);
  await page
    .getByRole("combobox", { name: "Logit lens token position", exact: true })
    .selectOption("0");
  await expect(
    page.getByText(
      "Some measurements at this position are unavailable in this capture.",
      { exact: true },
    ),
  ).toBeVisible();
  for (const [index, layer] of trace.layers.entries()) {
    const cells = page
      .locator(".position-lens-table tbody tr")
      .nth(index)
      .getByRole("cell");
    await expect(cells.nth(2)).toHaveText("N/A");
    await expect(cells.nth(3)).toHaveText(number(layer.token_entropies![0], 5));
  }
});

test("zero IDs and measurements preserve vocabulary identity when decoded prediction text is empty", async ({
  page,
}) => {
  await page.route("**/api/trace", async (route) => {
    const response = await route.fetch();
    const trace = await response.json();
    for (const layer of trace.layers) {
      layer.token_prediction_ids[0] = 0;
      layer.token_prediction_probabilities[0] = 1;
      layer.token_entropies[0] = 0;
      layer.token_norms[0] = 0;
      layer.token_predictions[0] = "";
    }
    await route.fulfill({ response, json: trace });
  });
  const trace = await capture(page);
  await expectPosition(page, trace, 0);
  await expect(
    page.locator(".position-lens-table .lens-token").first(),
  ).toHaveAttribute("title", "Vocabulary ID 0");
});

test("a new capture resets the position view to its own final prompt token", async ({
  page,
}) => {
  await capture(page);
  await page
    .getByRole("combobox", { name: "Logit lens token position", exact: true })
    .selectOption("0");
  await page.getByLabel("Prompt", { exact: true }).fill("the capital");
  const capturing = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  const trace: Trace = await (await capturing).json();
  await page.getByRole("tab", { name: "Logit lens", exact: true }).click();
  await expect(
    page.getByRole("combobox", {
      name: "Logit lens token position",
      exact: true,
    }),
  ).toHaveValue(String(trace.tokens.length - 1));
  await expectPosition(page, trace, trace.tokens.length - 1);
});
