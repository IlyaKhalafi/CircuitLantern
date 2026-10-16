"use client";

import {
  ArrowLeft,
  ArrowRight,
  Check,
  Download,
  FlaskConical,
  Plus,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ExperimentResult } from "@/components/experiment-result";
import { experimentArtifact, experimentLabel } from "@/lib/experiments";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import { api, downloadJSON } from "@/lib/api";
import { truncateText } from "@/lib/text";
import type { Experiment } from "@/lib/types";

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState<Experiment | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [name, setName] = useState("");
  const [saved, setSaved] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const selectedArtifact = selected ? experimentArtifact(selected) : undefined;
  useEffect(() => {
    api<Experiment[]>("/experiments")
      .then(setExperiments)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);
  async function open(id: string) {
    setBusy(true);
    setError(null);
    setSaved(false);
    setConfirmDelete(false);
    try {
      const result = await api<Experiment>(`/experiments/${id}`);
      setSelected(result);
      setNotes(result.notes);
      setName(result.name);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function save() {
    if (!selected || !name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api<Experiment>(`/experiments/${selected.id}`, {
        method: "PATCH",
        body: JSON.stringify({ name: name.trim(), notes }),
      });
      setSelected(result);
      setExperiments((items) =>
        items.map((item) => (item.id === result.id ? result : item)),
      );
      setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await api(`/experiments/${selected.id}`, { method: "DELETE" });
      setExperiments((items) =>
        items.filter((item) => item.id !== selected.id),
      );
      setSelected(null);
      setConfirmDelete(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="A RECORD OF YOUR CURIOSITY"
        title={selected ? selected.name : "Good questions deserve a notebook."}
        description={
          selected
            ? `${selected.model_id} · saved ${new Date(selected.created_at).toLocaleString()}`
            : "Save traces and sweeps, record observations, and return to the experiments that matter."
        }
        actions={
          selected ? (
            <button
              className="button secondary"
              onClick={() =>
                downloadJSON(
                  selected,
                  `experiment-${selected.id.slice(0, 8)}.json`,
                )
              }
            >
              <Download size={14} /> Export experiment
            </button>
          ) : (
            <Link className="button primary" href="/playground">
              <Plus size={14} /> New experiment
            </Link>
          )
        }
      />
      <ErrorNotice message={error} />
      {loading || (busy && !selected) ? (
        <Loading label="Opening experiment notebook" />
      ) : selected ? (
        <>
          <button
            className="button secondary"
            style={{ marginBottom: 20 }}
            onClick={() => setSelected(null)}
          >
            <ArrowLeft size={14} /> All experiments
          </button>
          <Panel
            title="Research notes"
            subtitle="The checkpoint revision and capture settings are preserved with every experiment."
          >
            <div className="experiment-detail">
              <label>
                Experiment name
                <input
                  className="form-input"
                  value={name}
                  onChange={(e) => {
                    setName(e.target.value);
                    setSaved(false);
                  }}
                  maxLength={100}
                />
              </label>
              <label style={{ marginTop: 17 }}>
                Observations
                <textarea
                  value={notes}
                  onChange={(e) => {
                    setNotes(e.target.value);
                    setSaved(false);
                  }}
                  placeholder="What changed? What would you test next?"
                  maxLength={10000}
                />
              </label>
              <div className="prompt-toolbar">
                <div style={{ display: "flex", gap: 10 }}>
                  <button
                    className="button secondary"
                    onClick={save}
                    disabled={busy || !name.trim()}
                  >
                    <Check size={14} />
                    {saved ? "Changes saved" : "Save notes"}
                  </button>
                  <button
                    className="button danger"
                    onClick={() =>
                      confirmDelete ? remove() : setConfirmDelete(true)
                    }
                    disabled={busy}
                  >
                    <Trash2 size={14} />
                    {confirmDelete ? "Confirm delete" : "Delete"}
                  </button>
                  {confirmDelete && (
                    <button
                      className="button secondary"
                      onClick={() => setConfirmDelete(false)}
                    >
                      Cancel
                    </button>
                  )}
                </div>
                <span
                  className="mono"
                  style={{ fontSize: 9, color: "var(--muted)" }}
                >
                  revision{" "}
                  {selectedArtifact?.revision?.slice(0, 10) ?? "unavailable"}
                </span>
              </div>
            </div>
          </Panel>
          <ExperimentResult key={selected.id} experiment={selected} />
        </>
      ) : experiments.length ? (
        <div className="experiment-grid">
          {experiments.map((exp) => (
            <button
              key={exp.id}
              className="panel experiment-card"
              style={{ textAlign: "left" }}
              onClick={() => open(exp.id)}
            >
              <span className="pill subtle">
                <FlaskConical size={11} /> {experimentLabel(exp.kind)}
              </span>
              <h3>{exp.name}</h3>
              <span className="mono">{exp.model_id}</span>
              <p style={{ marginTop: 13 }}>
                {exp.notes
                  ? truncateText(exp.notes, 110)
                  : "Open this experiment to explore its captured result and add research notes."}
              </p>
              <div className="experiment-meta">
                <span>{new Date(exp.created_at).toLocaleDateString()}</span>
                <ArrowRight size={14} />
              </div>
            </button>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<FlaskConical size={28} />}
          title="Start collecting evidence."
          href="/playground"
          action="Run your first trace"
        >
          Save an activation trace, residual patch sweep or head ablation sweep
          to keep its measurements and your observations together.
        </EmptyState>
      )}
    </>
  );
}
