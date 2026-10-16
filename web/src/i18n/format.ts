import { IntlMessageFormat } from "intl-messageformat";
import english from "./messages/en.json";
import type { Locale } from "./locale";
import type { Messages } from "./messages";

const compiled = new Map<string, IntlMessageFormat>();
function formatter(message: string, locale: Locale) {
  const key = locale + "\u0000" + message;
  let value = compiled.get(key);
  if (!value) {
    value = new IntlMessageFormat(message, locale, undefined, {
      ignoreTag: true,
    });
    if (compiled.size >= 1024) compiled.delete(compiled.keys().next().value!);
    compiled.set(key, value);
  }
  return value;
}
function escape(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
const patterns = Object.keys(english)
  .flatMap((source) => {
    const slots = [...source.matchAll(/\{(\w+)\}/g)];
    if (!slots.length || source.replace(/\{\w+\}/g, "").match(/[{}]/))
      return [];
    let cursor = 0,
      expression = "^";
    for (const slot of slots) {
      expression += escape(source.slice(cursor, slot.index)) + "([\\s\\S]*?)";
      cursor = slot.index! + slot[0].length;
    }
    expression += escape(source.slice(cursor)) + "$";
    const prefix = source.slice(0, slots[0].index);
    return prefix
      ? [
          {
            source,
            prefix,
            slots: slots.map((slot) => slot[1]),
            expression: new RegExp(expression),
          },
        ]
      : [];
  })
  .sort(
    (a, b) =>
      b.prefix.length - a.prefix.length || b.source.length - a.source.length,
  );

export function messageSource(source: string, messages: Messages) {
  if (Object.hasOwn(messages, source)) return { source, values: undefined };
  for (const pattern of patterns) {
    if (!source.startsWith(pattern.prefix)) continue;
    const match = source.match(pattern.expression);
    if (match)
      return {
        source: pattern.source,
        values: Object.fromEntries(
          pattern.slots.map((key, index) => [key, match[index + 1]]),
        ),
      };
  }
  return { source, values: undefined };
}
export function text(
  locale: Locale,
  messages: Messages,
  source: string,
  values?: Record<string, string | number | boolean | null | undefined>,
) {
  if (locale === "en" && !values) return source;
  const resolved = values
    ? { source, values }
    : messageSource(source, messages);
  const translated = Object.hasOwn(messages, resolved.source)
    ? messages[resolved.source]
    : resolved.source;
  if (!resolved.values) return translated;
  return String(formatter(translated, locale).format(resolved.values));
}
export function richMessage<T>(
  locale: Locale,
  messages: Messages,
  source: string,
  values: Record<string, T | string | number | boolean | null | undefined>,
) {
  return formatter(
    Object.hasOwn(messages, source) ? messages[source] : source,
    locale,
  ).format<T>(values);
}
