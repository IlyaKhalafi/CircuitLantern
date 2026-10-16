import type { Locale } from "./locale";
export type Messages = Record<string, string>;
export async function loadMessages(locale: Locale): Promise<Messages> {
  switch (locale) {
    case "fa":
      return (await import("./messages/fa.json")).default;
    case "ar":
      return (await import("./messages/ar.json")).default;
    case "ja":
      return (await import("./messages/ja.json")).default;
    case "fr":
      return (await import("./messages/fr.json")).default;
    default:
      return (await import("./messages/en.json")).default;
  }
}
