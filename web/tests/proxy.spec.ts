import { expect, test } from "@playwright/test";

test("the API proxy preserves validation errors and missing-resource status", async ({
  request,
}) => {
  const missing = await request.get("/api/jobs/missing-proxy-check");
  expect(missing.status()).toBe(404);
  expect((await missing.json()).detail).toContain("not found");
  const invalid = await request.post("/api/models/metadata", { data: {} });
  expect(invalid.status()).toBe(422);
  expect(Array.isArray((await invalid.json()).detail)).toBe(true);
});

test("the API proxy forwards JSON edits and empty delete responses", async ({
  request,
}) => {
  const traceResponse = await request.post("/api/trace", {
    data: {
      prompt: "the capital",
      max_new_tokens: 1,
      capture_attention: false,
    },
  });
  expect(traceResponse.ok()).toBe(true);
  const created = await request.post("/api/experiments", {
    data: {
      name: "Proxy transport verification",
      notes: "",
      trace: await traceResponse.json(),
    },
  });
  expect(created.status()).toBe(201);
  const experiment = await created.json();
  const edited = await request.patch(`/api/experiments/${experiment.id}`, {
    data: { notes: "π → projection" },
  });
  expect(edited.status()).toBe(200);
  expect((await edited.json()).notes).toBe("π → projection");
  const deleted = await request.delete(`/api/experiments/${experiment.id}`);
  expect(deleted.status()).toBe(204);
  expect(await deleted.text()).toBe("");
});

test("an unavailable backend shows a useful error without a false empty-search result", async ({
  page,
}) => {
  await page.route("**/api/**", (route) =>
    route.fulfill({
      status: 502,
      contentType: "application/json",
      body: JSON.stringify({
        detail:
          "Cannot reach the Python backend. Start it and check LANTERN_BACKEND_URL.",
      }),
    }),
  );
  await page.goto("/");
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "Cannot reach the Python backend",
  );
  await expect(
    page.getByText("Backend offline", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("No models found.", { exact: false }),
  ).not.toBeVisible();
  await page.screenshot({
    path: "../artifacts/screenshots/backend-unavailable.png",
    fullPage: true,
  });
});

test("a busy model can be retried without suggesting a replacement checkpoint", async ({
  page,
}) => {
  await page.route("**/api/graph", (route) =>
    route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({
        detail: "The model is busy with another operation.",
      }),
    }),
  );
  await page.goto("/workspace");
  await expect(page.getByRole("main").getByRole("alert")).toContainText("busy");
  await expect(
    page.getByRole("link", { name: "Choose a model" }),
  ).not.toBeVisible();
  await page.unroute("**/api/graph");
  await page.getByRole("button", { name: "Retry explorer" }).click();
  await expect(
    page.getByText("Model architecture", { exact: true }),
  ).toBeVisible();
});
