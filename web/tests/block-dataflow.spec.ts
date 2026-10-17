import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

async function hybridFixture(
  page: import("@playwright/test").Page,
  layers = 4,
) {
  const graph = await (
    await page.request.get(`/api/test/hybrid-graph?layers=${layers}`)
  ).json();
  await page.route("**/api/graph**", (route) => route.fulfill({ json: graph }));
  return graph;
}

test("hybrid layers show distinct connected mixer paths and exact matrix links", async ({
  page,
}) => {
  await hybridFixture(page);
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "Block dataflow", exact: true })
    .click();
  await expect(page.getByText("8 of 8 block matrices connected")).toBeVisible();
  await expect(
    page
      .locator(".flow-operation")
      .getByText("Gated DeltaNet recurrence", { exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator(".flow-operation")
      .getByText("Attention probabilities", { exact: true }),
  ).toHaveCount(0);
  await page
    .getByLabel("Diagram focus", { exact: true })
    .selectOption("attention");
  const matrixName = "model.layers.0.linear_attn.in_proj_qkv.weight";
  await page.route("**/api/matrix**", (route) =>
    route.fulfill({
      status: 400,
      json: { detail: "Weight matrix was not found in the loaded model." },
    }),
  );
  const request = page.waitForRequest(
    (request) =>
      request.url().includes("/api/matrix?") &&
      new URL(request.url()).searchParams.get("name") === matrixName,
  );
  await page
    .getByRole("button", { name: `Inspect ${matrixName}`, exact: true })
    .click();
  await request;
  await page
    .getByRole("button", { name: "Block 3 · Full attention", exact: true })
    .click();
  await expect(page.getByLabel("Focus on transformer block")).toHaveValue("3");
  await expect(page.getByText("7 of 7 block matrices connected")).toBeVisible();
  await expect(
    page
      .locator(".flow-operation")
      .getByText("Multiply output gate", { exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator(".flow-operation")
      .getByText("Gated DeltaNet recurrence", { exact: true }),
  ).toHaveCount(0);
  await capturePage(page, {
    path: "../artifacts/screenshots/hybrid-full-attention.png",
  });
});

test("connection tracing isolates branches and resets across layer changes", async ({
  page,
}) => {
  const graph = await hybridFixture(page);
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "Block dataflow", exact: true })
    .click();
  const update = graph.dataflow[0].nodes.find(
    (node: { label: string }) => node.label === "Update projection",
  );
  await page
    .getByLabel("Trace connections", { exact: true })
    .selectOption(update.id);
  await expect(page.locator(".flow-dimmed").first()).toBeVisible();
  await page.getByLabel("Trace connections", { exact: true }).focus();
  await page.keyboard.press("Home");
  await expect(
    page.getByLabel("Trace connections", { exact: true }),
  ).toHaveValue("");
  await expect(page.locator(".flow-dimmed")).toHaveCount(0);
  await page
    .getByLabel("Trace connections", { exact: true })
    .selectOption(update.id);
  await page
    .getByRole("button", { name: "Block 1 · Gated DeltaNet", exact: true })
    .click();
  await expect(
    page.getByLabel("Trace connections", { exact: true }),
  ).toHaveValue("");
  await expect(page.locator(".flow-dimmed")).toHaveCount(0);
});

test("summarized internals disclose incomplete coverage and retain inspectable weights", async ({
  page,
}) => {
  const graph = await hybridFixture(page);
  const flow = graph.dataflow[0];
  const weight = flow.nodes.find(
    (node: { weight: string }) => node.weight,
  )?.weight;
  flow.unmapped_matrices = [weight];
  flow.opaque_modules = ["model.layers.0.linear_attn"];
  flow.mapped_matrices -= 1;
  flow.nodes = flow.nodes.map((node: { weight: string }) =>
    node.weight === weight ? { ...node, weight: null, opaque: true } : node,
  );
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "Block dataflow", exact: true })
    .click();
  await expect(page.getByText("7 of 8 block matrices connected")).toBeVisible();
  await expect(
    page.getByText("Some module internals remain summarized"),
  ).toBeVisible();
  await page.getByText("Unmapped matrices", { exact: true }).click();
  await expect(
    page.getByRole("button", { name: weight, exact: true }),
  ).toBeVisible();
});

test("hybrid graph remains usable in both themes and on a narrow screen", async ({
  page,
}) => {
  await hybridFixture(page);
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "Block dataflow", exact: true })
    .click();
  await page
    .getByLabel("Diagram focus", { exact: true })
    .selectOption("attention");
  for (const theme of ["light", "dark"]) {
    await page
      .getByRole("button", {
        name: theme === "dark" ? "Dark theme" : "Light theme",
        exact: true,
      })
      .click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await capturePage(page, {
      path: `../artifacts/screenshots/hybrid-deltanet-${theme}.png`,
    });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
  await expect(
    page.getByRole("button", { name: "Block 3 · Full attention", exact: true }),
  ).toBeVisible();
  await expect
    .poll(
      async () =>
        (await page.locator(".flow-weight").first().boundingBox())?.width ?? 0,
    )
    .toBeGreaterThan(110);
  await capturePage(page, {
    path: "../artifacts/screenshots/hybrid-deltanet-mobile.png",
  });
  await page.locator(".block-dataflow").scrollIntoViewIfNeeded();
  const canvas = (await page.locator(".block-dataflow").boundingBox())!;
  const projection = page.locator(".flow-weight").first();
  const before = (await projection.boundingBox())!;
  await page.mouse.move(canvas.x + 50, canvas.y + 15);
  await page.mouse.down();
  await page.mouse.move(canvas.x + 300, canvas.y + 15, { steps: 12 });
  await page.mouse.up();
  await expect
    .poll(
      async () => ((await projection.boundingBox())?.x ?? before.x) - before.x,
    )
    .toBeGreaterThan(200);
});

test("unresolved layer kinds never masquerade as full attention", async ({
  page,
}) => {
  const graph = await hybridFixture(page);
  graph.dataflow[2].mixer_type = null;
  await page.goto("/workspace");
  await page
    .getByRole("button", { name: "Block dataflow", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Block 2 · Other mixer", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Block 2 · Full attention", exact: true }),
  ).toHaveCount(0);
});

test("larger source-derived architectures open directly on readable connections", async ({
  page,
}) => {
  await hybridFixture(page, 12);
  await page.goto("/workspace");
  await expect(
    page.getByRole("button", { name: "Block dataflow", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByLabel("Diagram focus", { exact: true })).toHaveValue(
    "attention",
  );
  await expect(page.getByText("8 of 8 block matrices connected")).toBeVisible();
  await expect
    .poll(
      async () =>
        (await page.locator(".flow-weight").first().boundingBox())?.width ?? 0,
    )
    .toBeGreaterThan(110);
  const matrix = await page.locator(".flow-weight").first().boundingBox();
  const canvas = await page.locator(".block-dataflow").boundingBox();
  expect(matrix!.x).toBeGreaterThan(canvas!.x);
  expect(matrix!.x + matrix!.width).toBeLessThan(canvas!.x + canvas!.width);
});

for (const locale of ["en", "fa", "ar", "ja", "fr"]) {
  test(`hybrid mixer descriptions and mobile controls remain usable in ${locale}`, async ({
    page,
    context,
  }) => {
    const { readFileSync } = await import("node:fs");
    const messages = JSON.parse(
      readFileSync(`src/i18n/messages/${locale}.json`, "utf8"),
    );
    await context.addCookies([
      {
        name: "lantern_locale",
        value: locale,
        url: new URL(test.info().project.use.baseURL as string).origin,
      },
    ]);
    await hybridFixture(page, 12);
    await page.goto("/workspace");
    await expect(page.locator("html")).toHaveAttribute("lang", locale);
    await expect(
      page
        .locator(".flow-operation")
        .getByText(messages["Gated DeltaNet recurrence"], { exact: true }),
    ).toBeVisible();
    await expect(
      page
        .locator(".flow-operation")
        .getByText(
          messages[
            "Chunked gated delta rule · recurrent state · not softmax attention"
          ],
          { exact: true },
        ),
    ).toBeVisible();
    await page.setViewportSize({ width: 390, height: 844 });
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      )
      .toBe(true);
    await page.locator(".graph-toolbar select").selectOption("3");
    await expect(
      page
        .locator(".flow-operation")
        .getByText(messages["Multiply output gate"], { exact: true }),
    ).toBeVisible();
    await expect(
      page
        .locator(".flow-operation")
        .getByText(messages["Attention output × sigmoid(gate)"], {
          exact: true,
        }),
    ).toBeVisible();
    await capturePage(page, {
      path: `../artifacts/screenshots/hybrid-locale-${locale}-mobile.png`,
    });
  });
}
