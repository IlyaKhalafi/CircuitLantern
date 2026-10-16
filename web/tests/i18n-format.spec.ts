import { expect, test } from "@playwright/test";
import assert from "node:assert/strict";

import { createElement } from "react";

import { IntlMessageFormat } from "intl-messageformat";

import en from "../src/i18n/messages/en.json";

import fa from "../src/i18n/messages/fa.json";

import ar from "../src/i18n/messages/ar.json";

import ja from "../src/i18n/messages/ja.json";

import fr from "../src/i18n/messages/fr.json";

import { interventionLabel } from "../src/lib/interventions";

import { richMessage, text } from "../src/i18n/format";

import { direction, resolveLocale } from "../src/i18n/locale";

function argumentsOf(message: IntlMessageFormat) {
  const names = new Map<string, boolean>();
  function visit(nodes: ReturnType<IntlMessageFormat["getAst"]>) {
    for (const node of nodes) {
      if (node.type >= 1 && node.type <= 6 && "value" in node)
        names.set(node.value, node.type === 6 || node.type === 2);
      if ("options" in node)
        for (const option of Object.values(node.options)) visit(option.value);
    }
  }
  visit(message.getAst());
  return names;
}

test("every catalog preserves message arguments and formats plural branches", () => {
  const originals = Object.entries(en).map(([key, source]) => ({
    key,
    args: argumentsOf(
      new IntlMessageFormat(source, "en", undefined, { ignoreTag: true }),
    ),
  }));
  const keys = Object.keys(en).sort();
  for (const [locale, messages] of Object.entries({ en, fa, ar, ja, fr })) {
    // These synchronous checks need no browser polling or trace step per message.
    assert.deepEqual(Object.keys(messages).sort(), keys, locale);
    for (const { key, args } of originals) {
      const translated = new IntlMessageFormat(
        (messages as Record<string, string>)[key],
        locale,
        undefined,
        { ignoreTag: true },
      );
      assert.deepEqual(
        [...argumentsOf(translated).keys()].sort(),
        [...args.keys()].sort(),
        locale + " " + key,
      );
      for (const count of [0, 1, 2, 6]) {
        const values = Object.fromEntries(
          [...args].map(([name, numeric]) => [
            name,
            numeric ? count : "RESEARCH_VALUE",
          ]),
        );
        assert.doesNotThrow(
          () => translated.format(values),
          locale + " " + key,
        );
      }
    }
  }
});

test("formatting reorders rich nodes and preserves literal research data", () => {
  const token = createElement("code", null, "Mary {x}");
  const result = richMessage(
    "fr",
    { "{a} then {b}": "{b} puis {a}" },
    "{a} then {b}",
    { a: token, b: "John" },
  );
  expect(Array.isArray(result)).toBe(true);
  expect(result).toContain(token);
  expect(result).toContain("John puis ");
  expect(text("fr", { "Layer {v0}": "Couche {v0}" }, "Layer 12")).toBe(
    "Couche 12",
  );
  expect(
    text("fr", { "Layer {v0}": "Couche {v0}" }, "Unrelated prompt Mary {x}"),
  ).toBe("Unrelated prompt Mary {x}");
  expect(text("en", {}, "can't {value}", { value: 0 })).toBe("can't 0");
  expect(
    text("en", {}, "{n, plural, one {# token} other {# tokens}}", { n: 2 }),
  ).toBe("2 tokens");
  for (const name of ["constructor", "toString", "__proto__"]) {
    expect(text("fr", {}, name)).toBe(name);
    expect(richMessage("fr", {}, name, {})).toBe(name);
  }
  expect(resolveLocale("unknown")).toBe("en");
  expect(direction("fa")).toBe("rtl");
  expect(direction("ar")).toBe("rtl");
  expect(direction("ja")).toBe("ltr");
});

test("intervention diagnostics resolve their complete message", () => {
  const head = { ablate_layer: null, ablate_head: { layer: 3, head: 5 } };
  const block = { ablate_layer: 3, ablate_head: null };
  expect(text("fr", fr, interventionLabel(head))).toBe(
    text("fr", fr, "Layer {v0} · head {v1} zeroed", { v0: 3, v1: 5 }),
  );
  expect(text("fr", fr, interventionLabel(block))).toBe(
    text("fr", fr, "Block {v0} bypassed", { v0: 3 }),
  );
  expect(
    text(
      "fr",
      fr,
      "The server returned 503. Check that the Python backend is running.",
    ),
  ).toBe(
    text(
      "fr",
      fr,
      "The server returned {status}. Check that the Python backend is running.",
      { status: 503 },
    ),
  );
});
