import type { Page } from "@playwright/test";

export async function capturePage(
  page: Page,
  options: Parameters<Page["screenshot"]>[0],
) {
  await page.waitForTimeout(100);
  await page.evaluate(async () => {
    await document.fonts.ready;
    if (document.activeElement instanceof HTMLElement)
      document.activeElement.blur();
    // Cancel in-flight inspector scrolling before capturing fixed page elements.
    for (let frame = 0; frame < 4; frame++) {
      window.scrollTo({ top: 0, left: 0, behavior: "instant" });
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
    }
  });
  return page.screenshot({ fullPage: true, ...options });
}
