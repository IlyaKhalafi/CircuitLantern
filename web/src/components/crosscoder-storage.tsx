"use client";
import { useEffect, useRef, useState } from "react";
import { Download, Trash2 } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { api, downloadJSON } from "@/lib/api";
import { ErrorNotice } from "@/components/ui";
import type {
  CaptureSummary,
  CrosscoderSummary,
  DamagedArtifact,
} from "@/lib/crosscoders";

export function CrosscoderStorage({
  captures,
  comparisons,
  blocked,
  onDeleted,
}: {
  captures: (CaptureSummary | DamagedArtifact)[];
  comparisons: (CrosscoderSummary | DamagedArtifact)[];
  blocked: boolean;
  onDeleted: (kind: string, identifier: string) => void;
}) {
  const { t } = useI18n();
  const [kind, setKind] = useState("capture");
  const [identifier, setIdentifier] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const choices = kind === "capture" ? captures : comparisons;
  const selected = choices.find((item) => item.id === identifier);
  const damaged =
    selected && "integrity_error" in selected ? selected.integrity_error : null;
  const path =
    "/crosscoders/" + (kind === "capture" ? "captures/" : "") + identifier;
  async function exportEvidence() {
    setWorking(true);
    setError(null);
    try {
      const value = await api(path + (kind === "capture" ? "" : "/export"));
      if (active.current)
        downloadJSON(value, kind + "-" + identifier + ".json");
    } catch (reason) {
      if (active.current) setError((reason as Error).message);
    } finally {
      if (active.current) setWorking(false);
    }
  }
  async function remove() {
    setWorking(true);
    setError(null);
    try {
      await api(path, { method: "DELETE" });
      if (active.current) {
        onDeleted(kind, identifier);
        setIdentifier("");
        setConfirming(false);
      }
    } catch (reason) {
      if (active.current) setError((reason as Error).message);
    } finally {
      if (active.current) setWorking(false);
    }
  }
  if (!captures.length && !comparisons.length) return null;
  return (
    <details className="feature-methods crosscoder-storage">
      <summary>{t("Manage retained data")}</summary>
      <div className="feature-form crosscoder-details">
        <p className="muted">
          {t(
            "Captures are stored on this server. Saved comparisons retain their source evidence after a capture is deleted.",
          )}
        </p>
        <div className="feature-form-grid">
          <label>
            {t("Artifact type")}
            <select
              aria-label={t("Artifact type")}
              value={kind}
              disabled={working || blocked}
              onChange={(event) => {
                setKind(event.target.value);
                setIdentifier("");
                setConfirming(false);
              }}
            >
              <option value="capture">{t("Activation captures")}</option>
              <option value="crosscoder">
                {t("Shared feature comparisons")}
              </option>
            </select>
          </label>
          <label>
            {t("Saved artifact")}
            <select
              aria-label={t("Saved artifact")}
              value={selected?.id || ""}
              disabled={working || blocked}
              onChange={(event) => {
                setIdentifier(event.target.value);
                setConfirming(false);
              }}
            >
              <option value="">{t("Select a saved artifact")}</option>
              {choices.map((item) => (
                <option key={item.id} value={item.id}>
                  {"integrity_error" in item
                    ? t("Integrity check failed") + " · " + item.id
                    : item.name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <ErrorNotice message={error || damaged} />
        {selected && (
          <>
            <div className="feature-toolbar">
              <button
                className="button secondary"
                disabled={working || blocked || !!damaged}
                onClick={exportEvidence}
              >
                <Download size={14} />
                {t("Export evidence")}
              </button>
              {!damaged && (
                <a
                  className="button secondary"
                  href={
                    "/api" +
                    path +
                    (kind === "capture" ? "/tensors" : "/weights")
                  }
                  download
                >
                  {t(
                    kind === "capture"
                      ? "Download tensors"
                      : "Download weights",
                  )}
                </a>
              )}
              <button
                className="button secondary"
                disabled={working || blocked}
                onClick={() => setConfirming(true)}
              >
                <Trash2 size={14} />
                {t("Delete saved artifact")}
              </button>
            </div>
            {confirming && (
              <div className="feature-plan">
                <p>
                  {t(
                    "This removes the selected artifact from the server. Export any evidence you want to keep first.",
                  )}
                </p>
                <strong dir="auto">{selected.name}</strong>
                <div className="header-actions crosscoder-details">
                  <button
                    className="button secondary"
                    disabled={working}
                    onClick={() => setConfirming(false)}
                  >
                    {t("Cancel")}
                  </button>
                  <button
                    className="button danger"
                    disabled={working || blocked}
                    onClick={remove}
                  >
                    {t("Confirm deletion")}
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </details>
  );
}
