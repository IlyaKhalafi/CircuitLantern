"use client";

import { Modal } from "@/components/modal";

import {
  ArrowRight,
  ArrowUpRight,
  Box,
  Check,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Download,
  ExternalLink,
  HardDrive,
  Info,
  Layers3,
  LoaderCircle,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
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
import type { Compatibility, Hardware, Job, ModelCard } from "@/lib/types";
import { ErrorNotice, Loading, Metric, PageHeader } from "@/components/ui";

export default function LibraryPage() {
  const router = useRouter();
  const [models, setModels] = useState<ModelCard[]>([]);
  const [hardware, setHardware] = useState<Hardware | null>(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [report, setReport] = useState<Compatibility | null>(null);
  const [busy, setBusy] = useState(false);
  const [searching, setSearching] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [device, setDevice] = useState("cpu");
  const [context, setContext] = useState(256);
  const [precision, setPrecision] = useState("auto");
  const [revision, setRevision] = useState("main");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestId = useRef(0);
  useEffect(() => {
    api<ModelCard[]>("/models")
      .then(setModels)
      .catch((e) => setError(e.message))
      .finally(() => setSearching(false));
    api<Hardware>("/hardware")
      .then(setHardware)
      .catch(() => {});
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);
  async function search(event: React.FormEvent) {
    event.preventDefault();
    setSearching(true);
    setError(null);
    try {
      setModels(
        await api<ModelCard[]>(`/models?q=${encodeURIComponent(query)}`),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  }
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
  }
  async function poll(id: string) {
    try {
      const result = await api<Job>(`/jobs/${id}`);
      setJob(result);
      if (result.status === "ready") {
        router.push("/workspace");
        return;
      }
      if (result.status === "failed") {
        setError(result.message);
        setBusy(false);
        return;
      }
      timer.current = setTimeout(() => poll(id), 1200);
    } catch (e) {
      setError(
        `${(e as Error).message} The server may still be preparing the model. Check Server & runtime before retrying.`,
      );
      setBusy(false);
      setJob(null);
    }
  }
  async function load() {
    if (!selected || !report) return;
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
      setJob(result);
      await poll(result.id);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
      setJob(null);
    }
  }
  const isDownloading = job && !["failed", "ready"].includes(job.status);
  return (
    <>
      <PageHeader
        eyebrow="YOUR NEXT DISCOVERY STARTS HERE"
        title="Open a model. Look closer."
        description="Find a model, check your hardware, and start exploring what makes it work."
        actions={
          <a href="/guide" className="button secondary">
            Quick start guide <ArrowUpRight size={15} />
          </a>
        }
      />
      <section className="welcome-banner">
        <div>
          <span className="pill orange">
            <Sparkles size={13} /> THE OPEN MODEL WORKBENCH
          </span>
          <h2>
            Big questions.
            <br />
            Meet the small details.
          </h2>
          <p>
            Trace a prediction through every layer. Explore weight geometry.
            <br className="desktop-break" /> Turn a black box into something you
            can reason about.
          </p>
          <div className="banner-points">
            <span>
              <Check size={14} /> Real model weights
            </span>
            <span>
              <Check size={14} /> Runs on your server
            </span>
            <span>
              <Check size={14} /> Reproducible experiments
            </span>
          </div>
        </div>
        <div className="circuit-illustration" aria-hidden="true">
          <div className="circuit-orbit orbit-one" />
          <div className="circuit-orbit orbit-two" />
          <div className="circuit-core">
            <Layers3 size={42} />
          </div>
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div key={i} className={`circuit-node node-${i}`}>
              <span />
              {["Wq", "Wk", "Wv", "MLP", "x", "y"][i]}
            </div>
          ))}
          <span className="illustration-label label-top">
            ATTENTION → UNDERSTANDING
          </span>
          <span className="illustration-label label-bottom">
            FOLLOW THE SIGNAL
          </span>
        </div>
      </section>
      <div className="section-heading">
        <div>
          <h2>
            Model library <span className="count-badge">Hugging Face</span>
          </h2>
          <p>
            Start small, inspect deeply. Or bring any compatible checkpoint.
          </p>
        </div>
        <div className="pill subtle">
          <ShieldCheck size={14} /> Compatibility checked before download
        </div>
      </div>
      <form className="search-row" onSubmit={search}>
        <div className="search-input">
          <Search size={18} />
          <input
            aria-label="Search Hugging Face models"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search Hugging Face models or enter a repository ID…"
          />
          <kbd>↵</kbd>
        </div>
        <button className="button secondary" disabled={searching}>
          <SlidersHorizontal size={16} />
          {searching ? "Searching" : "Search Hub"}
        </button>
        {query.includes("/") && (
          <button
            type="button"
            className="button primary"
            onClick={() => check(query)}
          >
            Check model <ArrowRight size={16} />
          </button>
        )}
      </form>
      {!selected && <ErrorNotice message={error} />}
      {searching ? (
        <Loading label="Finding models" />
      ) : (
        <div className="model-grid">
          {models.map((model) => (
            <button
              className={`model-card ${selected === model.id ? "selected" : ""}`}
              key={model.id}
              onClick={() => check(model.id)}
            >
              <div className="model-card-top">
                <span className={`model-icon ${model.color}`}>
                  <Box size={23} />
                </span>
                <span
                  className={`pill ${model.tag === "Recommended" ? "orange" : "subtle"}`}
                >
                  {model.tag}
                </span>
              </div>
              <h3>{model.name}</h3>
              <span className="model-owner">{model.id.split("/")[0]}</span>
              <p>{model.description}</p>
              <div className="model-card-footer">
                <div>
                  <span>{model.parameters}</span>
                  <span>{model.family}</span>
                </div>
                <ArrowUpRight size={17} />
              </div>
            </button>
          ))}
        </div>
      )}
      {!searching && !error && models.length === 0 && (
        <div className="empty-inline">
          No models found. Try a shorter search or enter a full repository ID.
        </div>
      )}
      <div className="library-bottom">
        <div>
          <Cpu size={21} />
          <div>
            <strong>Your hardware, taken into account.</strong>
            <p>
              Memory capacity is just the beginning. Architecture and runtime
              matter too.
            </p>
          </div>
        </div>
        <a className="text-link" href="/hardware">
          View server details <ArrowRight size={15} />
        </a>
      </div>
      <div className="how-it-works">
        {[
          {
            n: "01",
            title: "Choose a checkpoint",
            text: "Browse open models or use a Hub repository ID.",
          },
          {
            n: "02",
            title: "Check the fit",
            text: "Verify memory, hardware and runtime support.",
          },
          {
            n: "03",
            title: "Follow your curiosity",
            text: "Inspect matrices and watch predictions emerge.",
          },
        ].map((step) => (
          <div key={step.n}>
            <span>{step.n}</span>
            <div>
              <h3>{step.title}</h3>
              <p>{step.text}</p>
            </div>
          </div>
        ))}
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
            <span className="eyebrow">DEPLOYMENT CHECK</span>
            <button
              className="icon-button"
              aria-label="Close compatibility check"
              disabled={!!isDownloading}
              onClick={() => {
                setSelected(null);
                ++requestId.current;
              }}
            >
              <X size={19} />
            </button>
          </div>
          <h2 id="check-title">A good fit for your research?</h2>
          <a
            className="model-id"
            href={`https://huggingface.co/${selected}`}
            target="_blank"
            rel="noreferrer"
          >
            {selected} <ExternalLink size={13} />
          </a>
          <div className="form-grid">
            <label>
              Execution device
              <select
                aria-label="Execution device"
                disabled={!!isDownloading}
                value={device}
                onChange={(e) => {
                  setDevice(e.target.value);
                  check(selected, e.target.value);
                }}
              >
                <option value="cpu">CPU · system memory</option>
                <option value="cuda">CUDA · NVIDIA GPU</option>
              </select>
            </label>
            <label>
              Context allowance
              <select
                aria-label="Context allowance"
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
                      {n.toLocaleString()} tokens
                    </option>
                  ),
                )}
              </select>
            </label>
            <label>
              Execution format
              <select
                aria-label="Execution format"
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
                      ? executionLabel(value, {
                          quantization: report?.metadata.quantization,
                        })
                      : label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <details className="estimate-details revision-details">
            <summary>
              Checkpoint revision <ChevronRight size={14} />
            </summary>
            <p>
              Choose a branch, tag or full commit hash. Loading uses the exact
              commit resolved by the deployment check.
            </p>
            <div className="revision-controls">
              <label>
                Hub revision
                <input
                  className="form-input"
                  value={revision}
                  maxLength={128}
                  placeholder="main"
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
                Check revision
              </button>
            </div>
          </details>
          <ErrorNotice message={error} />
          {busy && !report && (
            <Loading label="Reading checkpoint metadata and checking this server" />
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
                      ? "Ready to explore on this server"
                      : "This configuration needs attention"}
                  </strong>
                  <p>
                    {report.compatible
                      ? "The checkpoint passes the current hardware and runtime checks."
                      : "Resolve the checks below before loading this checkpoint."}
                  </p>
                </div>
              </div>
              <div className="metrics-grid compact">
                <Metric
                  label="Model parameters"
                  value={number(report.metadata.parameters, 0)}
                />
                <Metric
                  label="Analysis memory"
                  value={bytes(report.required_bytes)}
                />
                <Metric
                  label={
                    report.reclaimable_bytes
                      ? "Replacement budget"
                      : "Available budget"
                  }
                  value={bytes(report.available_bytes)}
                />
                <Metric
                  label={
                    report.cached_bytes ? "Remaining download" : "Download size"
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
                  Includes execution weights, cache and inspection overhead
                </span>
                <span>{report.device.toUpperCase()}</span>
              </div>
              {!!report.reclaimable_bytes && (
                <p className="inline-note">
                  Includes {bytes(report.reclaimable_bytes)} released by
                  replacing the active model.
                </p>
              )}
              {!!report.cached_bytes && (
                <p className="inline-note">
                  {report.download_bytes === 0
                    ? "All checkpoint files are cached locally."
                    : `Reusing ${bytes(report.cached_bytes)} of checkpoint files from the local Hub cache.`}
                </p>
              )}
              {report.blockers.map((b, i) => (
                <div className="check-message blocked" key={i}>
                  <Info size={15} />
                  <span>{b}</span>
                </div>
              ))}
              <details className="estimate-details">
                <summary>
                  How this estimate works <ChevronRight size={14} />
                </summary>
                <p>
                  Source checkpoint{" "}
                  {bytes(report.source_bytes ?? report.download_bytes)}.{" "}
                  Execution format {report.runtime_precision}. Resident weights{" "}
                  {bytes(report.weight_bytes)}. Matrix workspace{" "}
                  {bytes(report.analysis_workspace_bytes || 0)}. Logit buffers{" "}
                  {bytes(report.logits_bytes || 0)}. MLP buffers{" "}
                  {bytes(report.activation_bytes || 0)}.
                </p>
                {report.warnings.map((w, i) => (
                  <p key={i}>{w}</p>
                ))}
                <p>
                  Disk available{" "}
                  {hardware ? bytes(hardware.disk_free_bytes) : "loading"}.
                  Checkpoint commit{" "}
                  <a
                    className="text-link"
                    href={`https://huggingface.co/${selected}/tree/${encodeURIComponent(report.metadata.revision)}`}
                    target="_blank"
                    rel="noreferrer"
                    title={report.metadata.revision}
                  >
                    {report.metadata.revision.slice(0, 10)}
                  </a>
                  .
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
                <span>{job.message}</span>
                <strong>{job.progress}%</strong>
              </div>
              <progress value={job.progress} max={100} />
            </div>
          )}
          <div className="dialog-footer">
            <span>
              <HardDrive size={14} /> Weights stay on your server
            </span>
            <button
              className="button primary"
              onClick={load}
              disabled={!report?.compatible || busy}
            >
              <Download size={16} />
              {isDownloading
                ? "Preparing model"
                : report?.cached_bytes && report.download_bytes === 0
                  ? "Load & explore"
                  : "Download & explore"}
              <ArrowRight size={16} />
            </button>
          </div>
        </Modal>
      )}
    </>
  );
}
