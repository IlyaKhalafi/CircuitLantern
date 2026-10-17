"use client";

import { useEffect, useRef, useState } from "react";
import { Download } from "lucide-react";
import { api } from "@/lib/api";
import { useI18n } from "@/i18n/provider";

export function ArtifactDownload({
  path,
  label,
  disabled = false,
  onError,
}: {
  path: string;
  label: string;
  disabled?: boolean;
  onError: (message: string | null) => void;
}) {
  const { t } = useI18n();
  const [checking, setChecking] = useState(false);
  const active = useRef(true);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
      pending.current?.abort();
    };
  }, []);
  useEffect(() => {
    if (disabled) pending.current?.abort();
  }, [disabled]);
  async function download() {
    if (pending.current || disabled) return;
    const controller = new AbortController();
    pending.current = controller;
    setChecking(true);
    onError(null);
    try {
      const prepared = await api<{ ticket?: unknown } | null>(path, {
        method: "POST",
        body: "{}",
        signal: controller.signal,
      });
      if (!active.current || controller.signal.aborted) return;
      if (
        typeof prepared?.ticket !== "string" ||
        !/^[a-f0-9]{64}$/.test(prepared.ticket)
      )
        throw new Error("The download could not be prepared. Try again.");
      const anchor = document.createElement("a");
      anchor.href = "/api/crosscoders/downloads/" + prepared.ticket;
      anchor.download = "";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    } catch (reason) {
      if (active.current && !controller.signal.aborted)
        onError((reason as Error).message);
    } finally {
      if (pending.current === controller) pending.current = null;
      if (active.current) setChecking(false);
    }
  }
  return (
    <button
      type="button"
      className="button secondary"
      disabled={disabled || checking}
      aria-busy={checking}
      onClick={download}
    >
      <Download size={14} />
      {t(checking ? "Checking download" : label)}
    </button>
  );
}
