"use client";

import { useEffect, useRef, type ReactNode } from "react";

export function Modal({
  children,
  titleId,
  locked = false,
  onClose,
}: {
  children: ReactNode;
  titleId: string;
  locked?: boolean;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const close = useRef(onClose);
  const opener = useRef<HTMLElement | null>(null);
  close.current = onClose;
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (!opener.current && document.activeElement instanceof HTMLElement)
      opener.current = document.activeElement;
    element.showModal();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      element.close();
      document.body.style.overflow = previousOverflow;
      queueMicrotask(() => {
        if (opener.current?.isConnected) opener.current.focus();
      });
    };
  }, []);
  return (
    <dialog
      ref={dialog}
      className="model-dialog"
      aria-labelledby={titleId}
      onCancel={(event) => {
        event.preventDefault();
        if (!locked) close.current();
      }}
      onClick={(event) => {
        if (event.target !== event.currentTarget || locked) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (
          event.clientX < bounds.left ||
          event.clientX > bounds.right ||
          event.clientY < bounds.top ||
          event.clientY > bounds.bottom
        )
          close.current();
      }}
    >
      {children}
    </dialog>
  );
}
