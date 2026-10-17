"use client";
import { FieldLabel } from "@/components/help-tooltip";
import { useI18n } from "@/i18n/provider";
import { Modal } from "@/components/modal";
import { ModelDiscovery } from "@/components/model-discovery";
import { matchingProbeHref } from "@/lib/induction-comparison";
import {
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Download,
  ExternalLink,
  HardDrive,
  Info,
  LoaderCircle,
  ShieldCheck,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, bytes, number, post } from "@/lib/api";
import {
  executionFormats,
  executionLabel,
  fourBitFormat,
} from "@/lib/execution";
import type { Compatibility, Hardware, Job } from "@/lib/types";
import { ErrorNotice, Loading, Metric, PageHeader } from "@/components/ui";
export default function LibraryPage() {
  const { t, rich } = useI18n();
  const router = useRouter();
  const [hardware, setHardware] = useState<Hardware | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [report, setReport] = useState<Compatibility | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | readonly string[] | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [device, setDevice] = useState("cpu");
  const [context, setContext] = useState(256);
  const [precision, setPrecision] = useState("auto");
  const [revision, setRevision] = useState("main");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestId = useRef(0);
  const checkQueue = useRef<Promise<void>>(Promise.resolve());
  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    if (
      query.get("return") === "induction" &&
      /^[a-f0-9]{32}$/.test(query.get("reference") || "")
    ) {
      const targetContext = Number(query.get("context"));
      if (
        Number.isSafeInteger(targetContext) &&
        targetContext >= 16 &&
        targetContext <= 131072
      )
        setContext(targetContext);
      const targetPrecision = query.get("precision");
      if (executionFormats.some(([value]) => value === targetPrecision))
        setPrecision(targetPrecision!);
      const targetDevice = query.get("device");
      if (targetDevice === "cpu" || targetDevice === "cuda")
        setDevice(targetDevice);
    }
    api<Hardware>("/hardware")
      .then(setHardware)
      .catch(() => {});
    return () => {
      ++requestId.current;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);
  async function check(
    modelId: string,
    targetDevice = device,
    targetContext = context,
    targetPrecision = precision,
    targetRevision = modelId === selected ? revision : "main",
  ) {
    const id = ++requestId.current;
    const requestedRevision = targetRevision.trim() || "main";
    setRevision(requestedRevision);
    setSelected(modelId);
    setReport(null);
    setError(null);
    setBusy(true);
    setJob(null);
    const perform = async () => {
      if (id !== requestId.current) return;
      try {
        const result = await post<Compatibility>("/models/compatibility", {
          model_id: modelId,
          revision: requestedRevision,
          device: targetDevice,
          context_length: targetContext,
          precision: targetPrecision,
        });
        if (id === requestId.current) setReport(result);
      } catch (e) {
        if (id === requestId.current) setError((e as Error).message);
      } finally {
        if (id === requestId.current) setBusy(false);
      }
    };
    const pending = checkQueue.current.then(perform, perform);
    checkQueue.current = pending;
    await pending;
  }
  async function poll(id: string, generation: number) {
    if (generation !== requestId.current) return;
    try {
      const result = await api<Job>(`/jobs/${id}`);
      if (generation !== requestId.current) return;
      setJob(result);
      if (result.status === "ready") {
        const query = new URLSearchParams(window.location.search);
        const reference = query.get("reference");
        if (
          query.get("return") === "induction" &&
          reference &&
          /^[a-f0-9]{32}$/.test(reference)
        ) {
          router.push(matchingProbeHref(reference));
          return;
        }
        router.push(
          new URLSearchParams(window.location.search).get("return") ===
            "compare"
            ? "/workspace?view=compare"
            : "/workspace",
        );
        return;
      }
      if (result.status === "failed") {
        setError(result.message);
        setBusy(false);
        return;
      }
      timer.current = setTimeout(() => poll(id, generation), 1200);
    } catch (e) {
      if (generation !== requestId.current) return;
      setError([
        (e as Error).message,
        "The server may still be preparing the model. Check Server & runtime before retrying.",
      ]);
      setBusy(false);
      setJob(null);
    }
  }
  async function load() {
    if (!selected || !report) return;
    const generation = requestId.current;
    setBusy(true);
    setError(null);
    setJob({
      id: "",
      model_id: selected,
      status: "queued",
      progress: 0,
      message: "Checking the pinned checkpoint before loading",
    });
    try {
      const result = await post<Job>("/models/load", {
        model_id: selected,
        revision: report.metadata.revision,
        precision: report.precision,
        context_length: report.context_length ?? context,
        device: report.device,
      });
      if (generation !== requestId.current) return;
      setJob(result);
      await poll(result.id, generation);
    } catch (e) {
      if (generation !== requestId.current) return;
      setError((e as Error).message);
      setBusy(false);
      setJob(null);
    }
  }
  const isDownloading = job && !["failed", "ready"].includes(job.status);
  return (
    <>
      <PageHeader
        eyebrow={t("YOUR NEXT DISCOVERY STARTS HERE")}
        title={t("Model library")}
        help={t(
          "Find a model, check your hardware, and start exploring what makes it work.",
        )}
        actions={
          <a href="/guide" className="button secondary">
            {t("Quick start guide")}
            <ArrowUpRight size={15} />
          </a>
        }
      />
      <div className="library-check-note">
        <ShieldCheck size={14} />
        {t("Compatibility checked before download")}
      </div>
      <ModelDiscovery onChoose={check} />
      {!selected && <ErrorNotice message={error} />}
      <div className="library-server-summary">
        <Cpu size={16} />
        {hardware && (
          <>
            <bdi>
              {hardware.execution_gpu?.name ||
                hardware.gpus[0]?.name ||
                t("CPU")}
            </bdi>
            {(hardware.execution_gpu || hardware.gpus[0]) && (
              <span>
                {rich("{amount} GPU free", {
                  amount: (
                    <bdi>
                      {bytes(
                        (hardware.execution_gpu || hardware.gpus[0]).free_bytes,
                      )}
                    </bdi>
                  ),
                })}
              </span>
            )}
            <span>
              {rich("{amount} RAM free", {
                amount: <bdi>{bytes(hardware.ram_available_bytes)}</bdi>,
              })}
            </span>
            <span>
              {rich("{amount} disk free", {
                amount: <bdi>{bytes(hardware.disk_free_bytes)}</bdi>,
              })}
            </span>
          </>
        )}
        <a className="text-link" href="/hardware">
          {t("View server details")}
          <ArrowRight size={14} />
        </a>
      </div>
      {selected && (
        <Modal
          titleId="check-title"
          locked={!!isDownloading}
          onClose={() => {
            setSelected(null);
            ++requestId.current;
          }}
        >
          <div className="dialog-header">
            <span className="eyebrow">{t("DEPLOYMENT CHECK")}</span>
            <button
              className="icon-button"
              aria-label={t("Close compatibility check")}
              disabled={!!isDownloading}
              onClick={() => {
                setSelected(null);
                ++requestId.current;
              }}
            >
              <X size={19} />
            </button>
          </div>
          <h2 id="check-title">{t("A good fit for your research?")}</h2>
          <a
            className="model-id"
            href={`https://huggingface.co/${selected}`}
            target="_blank"
            rel="noreferrer"
          >
            {selected} <ExternalLink size={13} />
          </a>
          <div className="form-grid">
            <div className="form-field">
              <FieldLabel
                id="deployment-device"
                label="Execution device"
                help="Select the CPU or a CUDA GPU exposed to this server."
              />
              <select
                id="deployment-device"
                aria-label={t("Execution device")}
                disabled={!!isDownloading}
                value={device}
                onChange={(e) => {
                  setDevice(e.target.value);
                  check(selected, e.target.value);
                }}
              >
                <option value="cpu">{t("CPU \u00B7 system memory")}</option>
                <option value="cuda">{t("CUDA \u00B7 NVIDIA GPU")}</option>
              </select>
            </div>
            <div className="form-field">
              <FieldLabel
                id="deployment-context"
                label="Context allowance"
                help="Maximum prompt and generated tokens kept in the loaded session. Larger contexts require more memory."
              />
              <select
                id="deployment-context"
                aria-label={t("Context allowance")}
                disabled={!!isDownloading}
                value={context}
                onChange={(e) => {
                  setContext(+e.target.value);
                  check(selected, device, +e.target.value);
                }}
              >
                {[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192].map(
                  (n) => (
                    <option key={n} value={n}>
                      {rich("{v0} tokens", {
                        v0: n.toLocaleString(),
                      })}
                    </option>
                  ),
                )}
              </select>
            </div>
            <div className="form-field">
              <FieldLabel
                id="deployment-precision"
                label="Execution format"
                help="Choose the weight format used by the runtime. Support depends on both the checkpoint and your hardware."
              />
              <select
                id="deployment-precision"
                aria-label={t("Execution format")}
                disabled={!!isDownloading}
                value={precision}
                onChange={(e) => {
                  setPrecision(e.target.value);
                  check(selected, device, context, e.target.value);
                }}
              >
                {executionFormats.map(([value, label]) => (
                  <option key={value} value={value}>
                    {value === "int4" &&
                    fourBitFormat(report?.metadata.quantization)
                      ? t(
                          executionLabel(value, {
                            quantization: report?.metadata.quantization,
                          }),
                        )
                      : t(label)}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <details className="estimate-details revision-details">
            <summary>
              {t("Checkpoint revision")}
              <ChevronRight size={14} />
            </summary>
            <p>
              {t(
                "Choose a branch, tag or full commit hash. Loading uses the exact commit resolved by the deployment check.",
              )}
            </p>
            <div className="revision-controls">
              <label>
                {t("Hub revision")}
                <input
                  dir="auto"
                  className="form-input"
                  value={revision}
                  maxLength={128}
                  placeholder={t("main")}
                  disabled={!!isDownloading}
                  autoComplete="off"
                  spellCheck={false}
                  onChange={(event) => {
                    setRevision(event.target.value);
                    ++requestId.current;
                    setReport(null);
                    setError(null);
                    setBusy(false);
                    setJob(null);
                  }}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      revision.trim() &&
                      !isDownloading
                    ) {
                      event.preventDefault();
                      check(selected);
                    }
                  }}
                />
              </label>
              <button
                type="button"
                className="button secondary"
                disabled={busy || !revision.trim()}
                onClick={() => check(selected)}
              >
                {t("Check revision")}
              </button>
            </div>
          </details>
          <ErrorNotice message={error} />
          {busy && !report && (
            <Loading
              label={t("Reading checkpoint metadata and checking this server")}
            />
          )}
          {report && (
            <>
              <div
                className={`compatibility-result ${report.compatible ? "ready" : "blocked"}`}
              >
                {report.compatible ? (
                  <CheckCircle2 size={23} />
                ) : (
                  <Info size={23} />
                )}
                <div>
                  <strong>
                    {report.compatible
                      ? t("Ready to explore on this server")
                      : t("This configuration needs attention")}
                  </strong>
                  <p>
                    {report.compatible
                      ? t(
                          "The checkpoint passes the current hardware and runtime checks.",
                        )
                      : t(
                          "Resolve the checks below before loading this checkpoint.",
                        )}
                  </p>
                </div>
              </div>
              <div className="metrics-grid compact">
                <Metric
                  label={t("Model parameters")}
                  value={number(report.metadata.parameters, 0)}
                />
                <Metric
                  label={t("Analysis memory")}
                  value={bytes(report.required_bytes)}
                />
                <Metric
                  label={
                    report.reclaimable_bytes
                      ? t("Replacement budget")
                      : t("Available budget")
                  }
                  value={bytes(report.available_bytes)}
                />
                <Metric
                  label={
                    report.cached_bytes
                      ? t("Remaining download")
                      : t("Download size")
                  }
                  value={bytes(report.download_bytes)}
                />
              </div>
              <div className="memory-bar">
                <div
                  style={{
                    width: `${Math.min(100, (report.required_bytes / Math.max(report.available_bytes, 1)) * 100)}%`,
                  }}
                />
              </div>
              <div className="memory-legend">
                <span>
                  {t(
                    "Includes execution weights, cache and inspection overhead",
                  )}
                </span>
                <span>{report.device.toUpperCase()}</span>
              </div>
              {!!report.reclaimable_bytes && (
                <p className="inline-note">
                  {rich(
                    "Includes {v0} released by replacing the active model.",
                    {
                      v0: bytes(report.reclaimable_bytes),
                    },
                  )}
                </p>
              )}
              {!!report.cached_bytes && (
                <p className="inline-note">
                  {report.download_bytes === 0
                    ? t("All checkpoint files are cached locally.")
                    : t(
                        "Reusing {v0} of checkpoint files from the local Hub cache.",
                        {
                          v0: bytes(report.cached_bytes),
                        },
                      )}
                </p>
              )}
              {report.blockers.map((b, i) => (
                <div className="check-message blocked" key={i}>
                  <Info size={15} />
                  <span>{t(b)}</span>
                </div>
              ))}
              <details className="estimate-details">
                <summary>
                  {t("How this estimate works")}
                  <ChevronRight size={14} />
                </summary>
                <p>
                  {rich(
                    "Source checkpoint {v0}. Execution format {v1}. Resident weights {v2}. Temporary workspace {v3}. Logit buffers {v4}. MLP buffers {v5}. Model buffers {v6}.",
                    {
                      v0: bytes(report.source_bytes ?? report.download_bytes),
                      v1: report.runtime_precision,
                      v2: bytes(report.weight_bytes),
                      v3: bytes(
                        report.workspace_bytes ??
                          report.analysis_workspace_bytes ??
                          0,
                      ),
                      v4: bytes(report.logits_bytes || 0),
                      v5: bytes(report.activation_bytes || 0),
                      v6: bytes(report.model_buffer_bytes ?? 0),
                    },
                  )}
                </p>
                {report.warnings.map((w, i) => (
                  <p key={i}>{t(w)}</p>
                ))}
                <p>
                  {rich("Disk available {v0}. Checkpoint commit {v1}.", {
                    v0: hardware
                      ? bytes(hardware.disk_free_bytes)
                      : t("loading"),
                    v1: (
                      <a
                        className="text-link"
                        href={`https://huggingface.co/${selected}/tree/${encodeURIComponent(report.metadata.revision)}`}
                        target="_blank"
                        rel="noreferrer"
                        title={report.metadata.revision}
                      >
                        {report.metadata.revision.slice(0, 10)}
                      </a>
                    ),
                  })}
                </p>
              </details>
            </>
          )}
          {job && (
            <div className="download-progress">
              <div>
                <LoaderCircle
                  size={16}
                  className={isDownloading ? "spin" : ""}
                />
                <span>{t(job.message)}</span>
                <strong>{job.progress}%</strong>
              </div>
              <progress value={job.progress} max={100} />
            </div>
          )}
          <div className="dialog-footer">
            <span>
              <HardDrive size={14} />
              {t("Weights stay on your server")}
            </span>
            <button
              className="button primary"
              onClick={load}
              disabled={!report?.compatible || busy}
            >
              <Download size={16} />
              {isDownloading
                ? t("Preparing model")
                : report?.cached_bytes && report.download_bytes === 0
                  ? t("Load & explore")
                  : t("Download & explore")}
              <ArrowRight size={16} />
            </button>
          </div>
        </Modal>
      )}
    </>
  );
}
