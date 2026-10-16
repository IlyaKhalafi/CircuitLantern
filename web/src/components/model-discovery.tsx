"use client";
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Box,
  Download,
  Heart,
  Search,
  SlidersHorizontal,
} from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { api, number } from "@/lib/api";
import type { ModelCard } from "@/lib/types";
import { ErrorNotice, Loading } from "@/components/ui";

export function ModelDiscovery({
  onChoose,
}: {
  onChoose: (id: string) => void;
}) {
  const { t, locale } = useI18n();
  const [models, setModels] = useState<ModelCard[]>([]);
  const [query, setQuery] = useState("");
  const [author, setAuthor] = useState("");
  const [search, setSearch] = useState({ query: "", author: "" });
  const [sort, setSort] = useState("recommended");
  const [days, setDays] = useState("0");
  const [likes, setLikes] = useState("0");
  const [limit, setLimit] = useState(24);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const sequence = useRef(0);
  useEffect(() => {
    const controller = new AbortController();
    const request = ++sequence.current;
    setLoading(true);
    setModels([]);
    setError(null);
    const params = new URLSearchParams({
      q: search.query,
      author: search.author,
      sort,
      days,
      min_likes: likes,
      limit: String(limit),
    });
    api<ModelCard[]>("/models?" + params, { signal: controller.signal })
      .then((value) => {
        if (request === sequence.current) setModels(value);
      })
      .catch((reason) => {
        if (!controller.signal.aborted && request === sequence.current)
          setError(reason.message);
      })
      .finally(() => {
        if (request === sequence.current) setLoading(false);
      });
    return () => {
      controller.abort();
      ++sequence.current;
    };
  }, [search, sort, days, likes, limit]);
  const filtered = days !== "0" || likes !== "0";
  return (
    <section aria-label={t("Find models")}>
      <form
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          if (!/^[a-zA-Z0-9_.-]*$/.test(author.trim())) {
            setError(
              t(
                "Use an author name such as HuggingFaceTB. Put repository IDs in the search field.",
              ),
            );
            return;
          }
          setLimit(24);
          setSearch({ query: query.trim(), author: author.trim() });
        }}
      >
        <div className="search-row">
          <div className="search-input">
            <Search size={18} />
            <input
              dir="auto"
              aria-label={t("Search Hugging Face models")}
              maxLength={180}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t(
                "Search Hugging Face models or enter a repository ID…",
              )}
            />
          </div>
          <button className="button secondary" type="submit">
            {t("Search Hub")}
          </button>
          {query.includes("/") && (
            <button
              type="button"
              className="button primary"
              onClick={() => onChoose(query.trim())}
            >
              {t("Check model")}
              <ArrowRight size={16} />
            </button>
          )}
        </div>
        <div className="discovery-controls">
          <label>
            {t("Sort models")}
            <select
              aria-label={t("Sort models")}
              value={sort}
              onChange={(event) => {
                setSort(event.target.value);
                setLimit(24);
              }}
            >
              {[
                ["recommended", "Recommended"],
                ["trending", "Trending"],
                ["downloads", "Most downloaded"],
                ["likes", "Most liked"],
                ["updated", "Recently updated"],
                ["newest", "Newest models"],
              ].map(([value, label]) => (
                <option key={value} value={value}>
                  {t(label)}
                </option>
              ))}
            </select>
          </label>
          <details className="discovery-filters">
            <summary>
              <SlidersHorizontal size={15} />
              {t("Filters")}
              {filtered && (
                <span className="count-badge">
                  {Number(days !== "0") + Number(likes !== "0")}
                </span>
              )}
            </summary>
            <div className="discovery-filter-fields">
              <label>
                {sort === "newest" ? t("Created within") : t("Updated within")}
                <select
                  aria-label={t("Model time window")}
                  value={days}
                  onChange={(event) => {
                    setDays(event.target.value);
                    setLimit(24);
                  }}
                >
                  {[
                    ["0", "Any time"],
                    ["7", "Last 7 days"],
                    ["30", "Last 30 days"],
                    ["90", "Last 90 days"],
                    ["365", "Last year"],
                  ].map(([value, label]) => (
                    <option key={value} value={value}>
                      {t(label)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                {t("Minimum likes")}
                <select
                  aria-label={t("Minimum likes")}
                  value={likes}
                  onChange={(event) => {
                    setLikes(event.target.value);
                    setLimit(24);
                  }}
                >
                  {[0, 10, 100, 1000].map((value) => (
                    <option key={value} value={value}>
                      {value === 0
                        ? t("Any")
                        : value.toLocaleString(locale) + "+"}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                {t("Author or organization")}
                <input
                  dir="ltr"
                  className="form-input"
                  aria-label={t("Author or organization")}
                  value={author}
                  onChange={(event) => setAuthor(event.target.value)}
                  maxLength={96}
                  pattern="[a-zA-Z0-9_.-]*"
                  placeholder={t("HuggingFaceTB")}
                />
              </label>
              <button className="button secondary" type="submit">
                {t("Apply filters")}
              </button>
              <button
                className="text-link"
                type="button"
                onClick={() => {
                  setDays("0");
                  setLikes("0");
                  setAuthor("");
                  setLimit(24);
                  setSearch({ query: query.trim(), author: "" });
                }}
              >
                {t("Clear filters")}
              </button>
            </div>
          </details>
          <span className="discovery-count" role="status">
            {loading
              ? t("Finding models")
              : t("{count} models shown", { count: models.length })}
          </span>
        </div>
      </form>
      {filtered && (
        <p className="inline-note">
          {t(
            "Time and likes filters search the first 500 Hub results in the selected order. Broaden filters or narrow the search if nothing matches.",
          )}
        </p>
      )}
      <ErrorNotice message={error} />
      {loading ? (
        <Loading label={t("Finding models")} />
      ) : (
        <div className="model-grid">
          {models.map((model) => (
            <button
              className="model-card"
              key={model.id}
              onClick={() => onChoose(model.id)}
            >
              <div className="model-card-top">
                <span className={"model-icon " + model.color}>
                  <Box size={23} />
                </span>
                <span
                  className={
                    "pill " +
                    (model.tag === "Recommended" ? "orange" : "subtle")
                  }
                >
                  {t(model.tag)}
                </span>
              </div>
              <h3>{model.name}</h3>
              <span className="model-owner">{model.id.split("/")[0]}</span>
              <p>{t(model.description)}</p>
              {(model.downloads != null || model.likes != null) && (
                <div className="model-popularity">
                  {model.downloads != null && (
                    <span
                      aria-label={t("{count} downloads", {
                        count: model.downloads,
                      })}
                    >
                      <Download size={13} />
                      {number(model.downloads, 0)}
                    </span>
                  )}
                  {model.likes != null && (
                    <span
                      aria-label={t("{count} likes", { count: model.likes })}
                    >
                      <Heart size={13} />
                      {number(model.likes, 0)}
                    </span>
                  )}
                  {model.updated_at && (
                    <time dateTime={model.updated_at}>
                      {new Intl.DateTimeFormat(locale, {
                        dateStyle: "medium",
                      }).format(new Date(model.updated_at))}
                    </time>
                  )}
                </div>
              )}
              <div className="model-card-footer">
                <div>
                  <span>{t(model.parameters)}</span>
                  <span>{model.family}</span>
                </div>
                <ArrowUpRight size={17} />
              </div>
            </button>
          ))}
        </div>
      )}
      {!loading && !error && !models.length && (
        <div className="empty-inline">
          {t(
            "No models found. Try a shorter search or enter a full repository ID.",
          )}
        </div>
      )}
      {!loading && !error && models.length === limit && limit < 96 && (
        <button
          className="button secondary"
          onClick={() => setLimit((value) => value + 24)}
        >
          {t("Show more models")}
        </button>
      )}
    </section>
  );
}
