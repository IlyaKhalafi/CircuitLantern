"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useExperiment } from "@/lib/use-experiment";
import { probeReferenceStorage } from "@/lib/induction-comparison";

export function useMatchingProbe() {
  const params = useSearchParams();
  const requested = params.get("reference");
  const [stored, setStored] = useState<string | null>(null);
  const [restored, setRestored] = useState(false);
  useEffect(() => {
    try {
      setStored(sessionStorage.getItem(probeReferenceStorage));
    } catch {}
    setRestored(true);
  }, []);
  const raw = requested ?? stored;
  const valid = raw === null || /^[a-f0-9]{32}$/.test(raw);
  const identifier = valid ? raw : null;
  const record = useExperiment(identifier);
  const result =
    record.selected?.kind === "induction_probe" ? record.selected.result : null;
  useEffect(() => {
    if (!requested || !valid) return;
    try {
      sessionStorage.setItem(probeReferenceStorage, requested);
    } catch {}
  }, [requested, valid]);
  const problem = !valid
    ? "The matching probe reference is invalid."
    : record.selected &&
        (!result || result.format !== "circuit-lantern-induction-probe-v2")
      ? "This saved probe lacks comparison provenance. Capture a new probe before repeating it across checkpoints."
      : null;
  function clear() {
    try {
      sessionStorage.removeItem(probeReferenceStorage);
    } catch {}
    setStored(null);
    const next = new URLSearchParams(window.location.search);
    next.delete("reference");
    window.history.replaceState(
      null,
      "",
      window.location.pathname + (next.size ? "?" + next : ""),
    );
  }
  return {
    identifier,
    record: record.selected,
    result,
    error: problem || record.error,
    loading: (!restored && requested === null) || record.loading,
    clear,
  };
}
