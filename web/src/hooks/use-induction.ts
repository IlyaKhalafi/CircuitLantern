"use client";

import { useEffect, useRef, useState } from "react";
import { api, post } from "@/lib/api";
import { useBackgroundJob } from "@/lib/jobs";
import { useRuntime } from "@/lib/runtime";
import type {
  InductionCapabilities,
  InductionPlan,
  InductionResult,
  InductionSettings,
} from "@/lib/induction";

function restoreSettings(
  execution: string,
  fallback: InductionSettings,
): InductionSettings {
  try {
    const value = JSON.parse(
      sessionStorage.getItem("lantern-induction-settings-" + execution) ||
        "null",
    );
    if (!value || Object.keys(fallback).some((key) => !(key in value)))
      return fallback;
    for (const key of [
      "block_length",
      "repeats",
      "examples",
      "seed",
      "causal_heads",
    ])
      if (!Number.isSafeInteger(value[key])) return fallback;
    if (
      typeof value.include_start_token !== "boolean" ||
      (value.layers !== null &&
        (!Array.isArray(value.layers) ||
          !value.layers.every(Number.isSafeInteger)))
    )
      return fallback;
    return value;
  } catch {
    return fallback;
  }
}

export function useInduction() {
  const model = useRuntime();
  const background = useBackgroundJob<InductionResult>(
    "lantern-induction-probe-job",
  );
  const execution = model.runtime?.metadata?.execution_id;
  const [capabilities, setCapabilities] =
    useState<InductionCapabilities | null>(null);
  const [draft, setDraft] = useState<{
    execution: string;
    value: InductionSettings;
  } | null>(null);
  const [plan, setPlan] = useState<InductionPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const pending = useRef(false);
  const mounted = useRef(true);
  const settings = draft && draft.execution === execution ? draft.value : null;
  const current =
    capabilities?.execution?.execution_id === execution ? capabilities : null;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (!execution || background.busy) return;
    const controller = new AbortController();
    setError(null);
    api<InductionCapabilities>("/induction/capabilities", {
      signal: controller.signal,
    })
      .then((value) => {
        if (controller.signal.aborted) return;
        setCapabilities(value);
        if (value.recommended && value.execution?.execution_id === execution) {
          setDraft((previous) =>
            previous?.execution === execution
              ? previous
              : {
                  execution,
                  value: restoreSettings(execution, value.recommended!),
                },
          );
        }
      })
      .catch((reason: Error) => {
        if (!controller.signal.aborted) setError(reason.message);
      });
    return () => controller.abort();
  }, [execution, background.busy, attempt]);
  useEffect(() => {
    setPlan(null);
  }, [execution]);
  function update(changes: Partial<InductionSettings>) {
    if (!settings || !execution || checking || background.busy) return;
    const value = { ...settings, ...changes };
    setDraft({ execution, value });
    setPlan(null);
    setError(null);
    try {
      sessionStorage.setItem(
        "lantern-induction-settings-" + execution,
        JSON.stringify(value),
      );
    } catch {}
  }
  async function check(run: boolean) {
    if (!settings || !execution || pending.current || background.busy) return;
    pending.current = true;
    setChecking(true);
    setError(null);
    background.clearError();
    const payload = { ...settings, expected_execution_id: execution };
    try {
      const value = await post<InductionPlan>("/induction/plan", payload);
      if (!mounted.current) return;
      setPlan(value);
      if (run && value.compatible)
        await background.start("/induction/probe", payload);
    } catch (reason) {
      if (mounted.current) setError((reason as Error).message);
    } finally {
      pending.current = false;
      if (mounted.current) setChecking(false);
    }
  }
  return {
    ...background,
    ...model,
    capabilities: current,
    settings,
    plan,
    update,
    check,
    error: model.error || error || background.error,
    busy: checking || background.busy,
    checking,
    retry: () => {
      setAttempt((value) => value + 1);
      model.refresh();
    },
  };
}
