import { expect, test } from "@playwright/test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";
import english from "../src/i18n/messages/en.json";

test("static interface copy stays in the translation catalog", () => {
  const missing: string[] = [];
  const raw: string[] = [];
  const allowed = new Set([
    "Hugging Face",
    "Circuit",
    "Lantern",
    "L",
    "v0.1.0",
  ]);
  function walk(directory: string) {
    for (const item of readdirSync(directory, { withFileTypes: true })) {
      const file = join(directory, item.name);
      if (item.isDirectory()) {
        if (item.name !== "i18n") walk(file);
        continue;
      }
      if (!/\.tsx?$/.test(file)) continue;
      const source = ts.createSourceFile(
        file,
        readFileSync(file, "utf8"),
        ts.ScriptTarget.Latest,
        true,
      );
      function visit(node: ts.Node) {
        if (
          ts.isJsxText(node) &&
          /[A-Za-z]/.test(node.text.trim()) &&
          !allowed.has(node.text.trim())
        )
          raw.push(file + " " + node.text.trim());
        if (
          ts.isJsxAttribute(node) &&
          ["aria-label", "title", "placeholder", "alt"].includes(
            node.name.getText(source),
          ) &&
          node.initializer &&
          ts.isStringLiteral(node.initializer) &&
          /[A-Za-z]/.test(node.initializer.text) &&
          !allowed.has(node.initializer.text)
        )
          raw.push(file + " " + node.initializer.text);
        if (ts.isStringLiteral(node)) {
          let expression: ts.Node = node;
          while (
            ts.isParenthesizedExpression(expression.parent) ||
            (ts.isConditionalExpression(expression.parent) &&
              (expression.parent.whenTrue === expression ||
                expression.parent.whenFalse === expression)) ||
            (ts.isBinaryExpression(expression.parent) &&
              [
                ts.SyntaxKind.BarBarToken,
                ts.SyntaxKind.QuestionQuestionToken,
              ].includes(expression.parent.operatorToken.kind))
          ) {
            expression = expression.parent;
          }
          const parent = expression.parent;
          const property = ts.isPropertyAssignment(parent)
            ? parent.name.getText(source)
            : "";
          const message =
            [
              "label",
              "title",
              "description",
              "subtitle",
              "text",
              "tag",
            ].includes(property) ||
            (ts.isCallExpression(parent) &&
              ["t", "rich", "localizedMetadata"].includes(
                parent.expression.getText(source),
              ) &&
              parent.arguments[0] === expression);
          if (message && node.text.trim() && !Object.hasOwn(english, node.text))
            missing.push(file + " " + node.text);
        }
        ts.forEachChild(node, visit);
      }
      visit(source);
    }
  }
  walk(join(process.cwd(), "src"));
  expect(raw).toEqual([]);
  expect(missing).toEqual([]);
  for (const label of ["Recommended", "Multilingual", "Experiments"])
    expect(Object.hasOwn(english, label), label).toBe(true);
});
