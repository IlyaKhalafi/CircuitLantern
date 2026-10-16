"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { RefreshCw, Sparkles } from "lucide-react";
import { ResearchNavigation } from "@/components/research-navigation";
import { useI18n } from "@/i18n/provider";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import { FeatureTraining } from "@/components/feature-training";
import { FeatureBrowser } from "@/components/feature-browser";
import { FeatureInspectionPanel } from "@/components/feature-inspection";
import { FeatureCircuits } from "@/components/feature-circuits";
import { api } from "@/lib/api";
import type {
  DictionaryManifest,
  DictionarySummary,
  FeatureCapabilities,
} from "@/lib/features";

export default function FeaturesPage() {
  const { t } = useI18n();
  const [capabilities, setCapabilities] = useState<FeatureCapabilities | null>(
    null,
  );
  const [dictionaries, setDictionaries] = useState<DictionarySummary[]>([]);
  const [selected, setSelected] = useState("");
  const [dictionary, setDictionary] = useState<DictionaryManifest | null>(null);
  const [feature, setFeature] = useState(0);
  const [tab, setTab] = useState("train");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reading, setReading] = useState(false);
  const sequence = useRef(0);
  const refresh = useCallback(async () => {
    const responses = await Promise.allSettled([
      api<FeatureCapabilities>("/features/capabilities"),
      api<DictionarySummary[]>("/features/dictionaries"),
    ]);
    if (responses[0].status === "fulfilled")
      setCapabilities(responses[0].value);
    else setError(responses[0].reason.message);
    if (responses[1].status === "fulfilled") {
      const listing = responses[1].value;
      setDictionaries(listing);
      setSelected((current) => current || listing[0]?.id || "");
    } else setError(responses[1].reason.message);
    setLoading(false);
  }, []);
  useEffect(() => {
    refresh();
  }, [refresh]);
  const trained = useCallback(
    (identifier?: string) => {
      void refresh();
      if (identifier) {
        setSelected(identifier);
        setTab("explore");
      }
    },
    [refresh],
  );
  useEffect(() => {
    if (!selected) {
      setDictionary(null);
      return;
    }
    const id = ++sequence.current;
    setReading(true);
    setError(null);
    setDictionary(null);
    api<DictionaryManifest>("/features/dictionaries/" + selected)
      .then((value) => {
        if (id === sequence.current) {
          setDictionary(value);
          setFeature(
            [...value.feature_statistics].sort(
              (a, b) => b.frequency - a.frequency || a.index - b.index,
            )[0]?.index || 0,
          );
        }
      })
      .catch((reason) => {
        if (id === sequence.current) setError(reason.message);
      })
      .finally(() => {
        if (id === sequence.current) setReading(false);
      });
    return () => {
      sequence.current++;
    };
  }, [selected]);
  async function updated(identifier: string) {
    const id = sequence.current;
    const value = await api<DictionaryManifest>(
      "/features/dictionaries/" + identifier,
    );
    if (id === sequence.current) {
      setDictionary((current) =>
        current?.id === identifier && value.id === identifier ? value : current,
      );
    }
  }
  const executionId = capabilities?.execution?.execution_id;
  return (
    <>
      <PageHeader
        eyebrow={t("SPARSE FEATURES")}
        title={t("From activations to testable features.")}
        description={t(
          "Train sparse autoencoders, inspect their features and test causal effects without writing code.",
        )}
        actions={
          <button
            className="button secondary"
            onClick={() => {
              setError(null);
              refresh();
            }}
          >
            <RefreshCw size={14} />
            {t("Refresh")}
          </button>
        }
      />
      <ResearchNavigation />
      <ErrorNotice message={error} />
      <div
        className="feature-tabs"
        role="tablist"
        aria-label={t("Feature lab workflow")}
        onKeyDown={(event) => {
          const values = ["train", "explore", "circuits"];
          const current = values.indexOf(tab);
          const rtl = document.documentElement.dir === "rtl";
          const next =
            event.key === "Home"
              ? 0
              : event.key === "End"
                ? 2
                : event.key === "ArrowRight"
                  ? (current + (rtl ? 2 : 1)) % 3
                  : event.key === "ArrowLeft"
                    ? (current + (rtl ? 1 : 2)) % 3
                    : null;
          if (next !== null) {
            event.preventDefault();
            setTab(values[next]);
            document
              .getElementById("feature-tab-button-" + values[next])
              ?.focus();
          }
        }}
      >
        {[
          ["train", "Train dictionaries"],
          ["explore", "Explore features"],
          ["circuits", "Feature circuits"],
        ].map(([value, label]) => (
          <button
            role="tab"
            key={value}
            aria-selected={tab === value}
            tabIndex={tab === value ? 0 : -1}
            aria-controls={"feature-tab-" + value}
            id={"feature-tab-button-" + value}
            onClick={() => setTab(value)}
          >
            {t(label)}
          </button>
        ))}
      </div>
      {loading && <Loading />}
      <div
        role="tabpanel"
        id="feature-tab-train"
        aria-labelledby="feature-tab-button-train"
        hidden={tab !== "train"}
      >
        {!loading && !capabilities?.supported && (
          <EmptyState
            icon={<Sparkles />}
            title={t("Load a model to train sparse dictionaries")}
            href="/"
            action="Open model library"
          >
            {t(
              "Saved dictionaries can still be explored without a loaded model.",
            )}
          </EmptyState>
        )}
        <FeatureTraining capabilities={capabilities} onTrained={trained} />
      </div>
      <div
        role="tabpanel"
        id="feature-tab-explore"
        aria-labelledby="feature-tab-button-explore"
        hidden={tab !== "explore"}
      >
        <Panel title={t("Saved dictionaries")}>
          <div className="feature-form">
            <label>
              {t("Choose a dictionary")}
              <select
                aria-label={t("Choose a dictionary")}
                value={selected}
                onChange={(e) => setSelected(e.target.value)}
              >
                <option value="">{t("Select a dictionary")}</option>
                {dictionaries.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} · {t("Layer {v0}", { v0: item.layer })}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </Panel>
        {reading && <Loading />}
        {!dictionaries.length && !loading && (
          <EmptyState
            icon={<Sparkles />}
            title={t("No sparse dictionaries yet")}
          >
            {t(
              "Train a dictionary to inspect activating examples and test feature interventions.",
            )}
          </EmptyState>
        )}
        {dictionary && (
          <>
            <FeatureBrowser
              key={dictionary.id}
              dictionary={dictionary}
              feature={feature}
              onSelect={setFeature}
              onUpdated={updated}
              onDeleted={(identifier) => {
                setDictionary((current) =>
                  current?.id === identifier ? null : current,
                );
                setSelected((current) =>
                  current === identifier ? "" : current,
                );
                void refresh();
              }}
            />
            <FeatureInspectionPanel
              key={dictionary.id + "-prompt"}
              dictionary={dictionary}
              feature={feature}
              executionId={executionId}
              enabled={!!capabilities?.supported}
            />
          </>
        )}
      </div>
      <div
        role="tabpanel"
        id="feature-tab-circuits"
        aria-labelledby="feature-tab-button-circuits"
        hidden={tab !== "circuits"}
      >
        <FeatureCircuits
          dictionaries={dictionaries}
          executionId={executionId}
          enabled={!!capabilities?.supported}
        />
      </div>
      <details className="feature-methods">
        <summary>{t("Methods and research papers")}</summary>
        <p>
          {t(
            "This interactive implementation adapts published SAE methods to small local corpora. It uses raw residual activations, a training-only RMS scale, document-separated validation and unit decoder directions.",
          )}
        </p>
        <ul>
          {capabilities?.references.map((reference) => (
            <li key={reference.url}>
              <a href={reference.url} target="_blank" rel="noreferrer">
                {reference.title}
              </a>{" "}
              · {reference.year}
            </li>
          ))}
        </ul>
        <p>
          {t(
            "ReLU neuron resampling and exponential weight averaging are disabled. TopK uses a shorter inactivity window for interactive training. Exact settings are recorded in each export.",
          )}
        </p>
      </details>
    </>
  );
}
