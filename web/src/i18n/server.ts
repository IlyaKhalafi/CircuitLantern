import { cookies } from "next/headers";
import { resolveLocale, localeCookie } from "./locale";
import { loadMessages } from "./messages";
import { text } from "./format";
export async function requestLanguage() {
  const locale = resolveLocale((await cookies()).get(localeCookie)?.value);
  return { locale, messages: await loadMessages(locale) };
}
export async function localizedMetadata(title: string) {
  const { locale, messages } = await requestLanguage();
  return { title: text(locale, messages, title) };
}
