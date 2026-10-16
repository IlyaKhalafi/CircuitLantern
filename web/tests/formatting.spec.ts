import { expect, test } from "@playwright/test";
import { number } from "../src/lib/api";

test("scientific values preserve sign and scale without overflowing statistic cells", () => {
  for (const value of [1e-30, -1e-30, 1e18, -1e18]) {
    const text = number(value, 5);
    expect(Math.abs(Number(text) / value - 1)).toBeLessThan(1e-5);
    expect(text.length).toBeLessThan(20);
  }
});
