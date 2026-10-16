"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  cloneElement,
  isValidElement,
  type ReactNode,
} from "react";
import { direction, localeCookie, type Locale } from "./locale";
import { loadMessages, type Messages } from "./messages";
import { richMessage, text } from "./format";

interface I18n {
  locale: Locale;
  pending: boolean;
  error: string | null;
  setLocale: (locale: Locale) => Promise<void>;
  t: (
    message: string,
    values?: Record<string, string | number | boolean | null | undefined>,
  ) => string;
  rich: (message: string, values: Record<string, ReactNode>) => ReactNode;
  localize: (value: ReactNode) => ReactNode;
  dateTime: (value: string | number | Date) => string;
}
const Context = createContext<I18n | null>(null);

export function I18nProvider({
  initialLocale,
  initialMessages,
  children,
}: {
  initialLocale: Locale;
  initialMessages: Messages;
  children: ReactNode;
}) {
  const [current, setCurrent] = useState({
    locale: initialLocale,
    messages: initialMessages,
  });
  const [pending, setPending] = useState(false);
  const sequence = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const t = useCallback<I18n["t"]>(
    (message, values) =>
      text(current.locale, current.messages, message, values),
    [current],
  );
  const setLocale = useCallback(async (locale: Locale) => {
    const request = ++sequence.current;
    setPending(true);
    setError(null);
    try {
      const messages = await loadMessages(locale);
      if (request !== sequence.current) return;
      document.cookie =
        localeCookie +
        "=" +
        locale +
        "; Path=/; Max-Age=31536000; SameSite=Lax" +
        (location.protocol === "https:" ? "; Secure" : "");
      document.documentElement.lang = locale;
      document.documentElement.dir = direction(locale);
      setCurrent({ locale, messages });
    } catch {
      if (request === sequence.current)
        setError("Language files could not be loaded. Try again.");
    } finally {
      if (request === sequence.current) setPending(false);
    }
  }, []);
  const value = useMemo<I18n>(
    () => ({
      locale: current.locale,
      pending,
      error,
      setLocale,
      t,
      rich(message, values) {
        const keyed = Object.fromEntries(
          Object.entries(values).map(([key, value]) => [
            key,
            isValidElement(value) ? cloneElement(value, { key }) : value,
          ]),
        );
        return richMessage<ReactNode>(
          current.locale,
          current.messages,
          message,
          keyed,
        );
      },
      localize(value) {
        return typeof value === "string" ? t(value) : value;
      },
      dateTime(value) {
        return new Intl.DateTimeFormat(current.locale, {
          dateStyle: "medium",
          timeStyle: "short",
        }).format(new Date(value));
      },
    }),
    [current, pending, error, setLocale, t],
  );
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useI18n() {
  const context = useContext(Context);
  if (!context) throw new Error("Language context is unavailable.");
  return context;
}
