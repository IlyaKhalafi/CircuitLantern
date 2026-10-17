"use client";

import { useEffect, useRef, useState } from "react";
import { api, post } from "@/lib/api";
import { useBackgroundJob } from "@/lib/jobs";
import { useRuntime } from "@/lib/runtime";
import { normalizeProbeDraft } from "@/lib/induction-settings";
import { useMatchingProbe } from "@/hooks/use-matching-probe";
import type {
  InductionCapabilities,
  InductionPlan,
  InductionResult,
  InductionSettings,
} from "@/lib/induction";

function restoreSettings(
  execution: string,
  fallback: InductionSettings,
  layers: number[],
  vocabularyPool: number,
): InductionSettings {
  try {
    return normalizeProbeDraft(
      JSON.parse(
        sessionStorage.getItem("lantern-induction-settings-" + execution) ||
          "null",
      ),
      fallback,
      layers,
      vocabularyPool,
    );
  } catch {
    return { ...fallback };
  }
}

export function useInduction() {
  const model = useRuntime();
  const background = useBackgroundJob<InductionResult>(
    "lantern-induction-probe-job",
  );
  const reference = useMatchingProbe();
  const execution = model.runtime?.metadata?.execution_id;
  const [capabilities, setCapabilities] =
    useState<InductionCapabilities | null>(null);
  const [draft, setDraft] = useState<{
    execution: string;
    reference: string | null;
    value: InductionSettings;
  } | null>(null);
  const [plan, setPlan] = useState<InductionPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const pending = useRef(false);
  const mounted = useRef(true);
  const generation = useRef(0);
  const current =
    capabilities?.execution?.execution_id === execution ? capabilities : null;
  const settings = draft && draft.execution === execution ? draft.value : null;
  const matchingReady =
    !reference.loading &&
    !reference.error &&
    draft?.reference === reference.identifier;
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
        if (!controller.signal.aborted) setCapabilities(value);
      })
      .catch((reason: Error) => {
        if (!controller.signal.aborted) setError(reason.message);
      });
    return () => controller.abort();
  }, [execution, background.busy, attempt]);
  useEffect(() => {
    if (
      !execution ||
      !current?.recommended ||
      reference.loading ||
      reference.error
    )
      return;
    const recorded = reference.result;
    if (reference.identifier && !recorded) return;
    setDraft((previous) => {
      if (
        previous?.execution === execution &&
        previous.reference === reference.identifier
      )
        return previous;
      let value =
        previous?.execution === execution && previous.reference === null
          ? previous.value
          : restoreSettings(
              execution,
              current.recommended!,
              current.layers.map((row) => row.layer),
              current.vocabulary_pool_size || 64,
            );
      if (recorded) {
        const layers = recorded.layer_targets.map((row) => row.layer);
        const all =
          current.layers.length === layers.length &&
          current.layers.every((row) => layers.includes(row.layer));
        value = {
          ...recorded.dataset.config,
          causal_heads: recorded.requested_causal_heads,
          include_start_token: recorded.dataset.start_token_id !== null,
          layers: all ? null : layers,
        };
      }
      return { execution, reference: reference.identifier, value };
    });
  }, [
    execution,
    current,
    reference.identifier,
    reference.result,
    reference.loading,
    reference.error,
  ]);
  useEffect(() => {
    if (!draft || draft.execution !== execution || draft.reference !== null)
      return;
    try {
      sessionStorage.setItem(
        "lantern-induction-settings-" + execution,
        JSON.stringify(draft.value),
      );
    } catch {}
  }, [draft, execution]);
  useEffect(() => {
    generation.current++;
    setPlan(null);
    setError(null);
  }, [execution, reference.identifier]);
  function update(changes: Partial<InductionSettings>) {
    if (
      !settings ||
      !execution ||
      checking ||
      background.busy ||
      reference.identifier
    )
      return;
    setDraft({
      execution,
      reference: null,
      value: { ...settings, ...changes },
    });
    setPlan(null);
    setError(null);
  }
  async function check(run: boolean) {
    if (
      !settings ||
      !execution ||
      !matchingReady ||
      pending.current ||
      background.busy
    )
      return;
    pending.current = true;
    const requestedGeneration = generation.current;
    setChecking(true);
    setError(null);
    background.clearError();
    const payload = {
      ...settings,
      expected_execution_id: execution,
      reference_experiment_id: reference.identifier,
    };
    try {
      const value = await post<InductionPlan>("/induction/plan", payload);
      if (!mounted.current || requestedGeneration !== generation.current)
        return;
      setPlan(value);
      if (run && value.compatible)
        await background.start("/induction/probe", payload);
    } catch (reason) {
      if (mounted.current && requestedGeneration === generation.current)
        setError((reason as Error).message);
    } finally {
      pending.current = false;
      if (mounted.current) setChecking(false);
    }
  }
  return {
    ...background,
    ...model,
    reference,
    capabilities: current,
    settings,
    plan,
    update,
    check,
    matchingReady,
    error: model.error || reference.error || error || background.error,
    busy: checking || background.busy || reference.loading,
    checking,
    retry: () => {
      setAttempt((value) => value + 1);
      model.refresh();
    },
  };
}
