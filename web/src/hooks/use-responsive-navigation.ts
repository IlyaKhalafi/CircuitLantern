"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type FocusEvent,
  type PointerEvent,
} from "react";

export function useResponsiveNavigation() {
  const [open, setOpen] = useState(false);
  const [mobile, setMobile] = useState(false);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const lastNavigationFocus = useRef<HTMLElement | null>(null);
  const resizeFocus = useRef<HTMLElement | null>(null);

  const isNavigationElement = useCallback(
    (element: Element) =>
      Boolean(
        toggleRef.current?.contains(element) ||
        element.closest("#workbench-navigation"),
      ),
    [],
  );
  const onFocusCapture = useCallback(
    (event: FocusEvent<HTMLElement>) => {
      lastNavigationFocus.current = isNavigationElement(event.target)
        ? event.target
        : null;
    },
    [isNavigationElement],
  );
  const onPointerDownCapture = useCallback(
    (event: PointerEvent<HTMLElement>) => {
      if (
        event.target instanceof Element &&
        !isNavigationElement(event.target)
      ) {
        lastNavigationFocus.current = null;
      }
    },
    [isNavigationElement],
  );
  const closeNavigation = useCallback(() => {
    setOpen(false);
    if (mobile) toggleRef.current?.focus();
  }, [mobile]);
  const toggleNavigation = useCallback(() => setOpen((value) => !value), []);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 720px)");
    const syncNavigation = () => {
      const active =
        document.activeElement === document.body
          ? lastNavigationFocus.current
          : document.activeElement;
      resizeFocus.current = null;
      if (document.hasFocus()) {
        if (query.matches && active?.closest("#workbench-navigation")) {
          resizeFocus.current = toggleRef.current;
        } else if (!query.matches && active === toggleRef.current) {
          resizeFocus.current = document.querySelector<HTMLElement>(
            "#workbench-navigation .brand",
          );
        }
      }
      setMobile(query.matches);
      if (!query.matches) setOpen(false);
    };
    syncNavigation();
    query.addEventListener("change", syncNavigation);
    return () => query.removeEventListener("change", syncNavigation);
  }, []);
  useLayoutEffect(() => {
    const target = resizeFocus.current;
    resizeFocus.current = null;
    const active = document.activeElement;
    if (
      target?.isConnected &&
      (active === document.body ||
        (active instanceof Element && isNavigationElement(active)))
    ) {
      target.focus();
    }
  }, [mobile, isNavigationElement]);

  useEffect(() => {
    if (!open) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeNavigation();
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [open, closeNavigation]);
  useEffect(() => {
    if (!mobile || !open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [mobile, open]);

  return {
    open,
    mobile,
    toggleRef,
    closeNavigation,
    toggleNavigation,
    onFocusCapture,
    onPointerDownCapture,
  };
}
