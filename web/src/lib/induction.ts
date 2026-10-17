import type { AttentionHeadTarget, ExecutionProvenance } from "@/lib/types";

export interface InductionSettings {
  block_length: number;
  repeats: number;
  examples: number;
  seed: number;
  causal_heads: number;
  include_start_token: boolean;
  layers: number[] | null;
}

export interface InductionCapabilities {
  loaded: boolean;
  supported: boolean;
  execution:
    (ExecutionProvenance & { model_id: string; revision: string }) | null;
  layers: AttentionHeadTarget[];
  recommended: InductionSettings | null;
  reason?: string;
  token_limit?: number;
  start_token_id?: number | null;
  vocabulary_pool_size?: number;
}

export interface InductionPlan {
  compatible: boolean;
  blockers: string[];
  estimated_host_bytes: number;
  estimated_gpu_bytes: number;
  estimated_disk_bytes: number;
  baseline_cache_bytes: number;
  tokens_per_probe: number;
  examples: number;
  selected_heads: number;
  causal_heads: number;
  requested_forward_passes: number;
  dataset_fingerprint?: string;
}

export interface InductionHead {
  layer: number;
  head: number;
  prefix_attention: number;
  prefix_std: number;
  duplicate_attention: number;
  previous_attention: number;
  representative_attention: number[];
  prefix_by_probe?: number[];
  duplicate_by_probe?: number[];
  previous_by_probe?: number[];
}

export interface InductionEffect {
  layer: number;
  head: number;
  requested_examples: number;
  completed_examples: number;
  complete: boolean;
  mean_delta_log_probability: number;
  mean_kl_divergence: number;
  examples: {
    probe: number;
    baseline_log_probability: number;
    ablated_log_probability: number;
    delta_log_probability: number;
    kl_divergence: number;
  }[];
}

export interface InductionResult extends ExecutionProvenance {
  model_id: string;
  revision: string;
  device: string;
  format: string;
  reference_experiment_id?: string;
  causal_candidate_policy?: string;
  input_policy?: string;
  active_tokenizer_special_ids?: number[];
  complete: boolean;
  cancelled?: boolean;
  error: string | null;
  failed_at: {
    stage: string;
    probe: number;
    layer?: number;
    head?: number;
  } | null;
  dataset_fingerprint: string;
  dataset: {
    config: Pick<
      InductionSettings,
      "block_length" | "repeats" | "examples" | "seed"
    >;
    start_token_id: number | null;
    vocabulary_pool_size: number;
    probes: {
      token_ids: number[];
      target_ids: number[];
      first_queries: number[];
      repeated_queries: number[];
    }[];
  };
  layer_targets: AttentionHeadTarget[];
  requested_examples: number;
  measured_examples: number;
  requested_causal_heads: number;
  completed_causal_heads: number;
  representative_probe: number | null;
  representative_query_position: number;
  representative_tokens: string[];
  heads: InductionHead[];
  probe_metrics: { probe: number; repeat_nll: number[] }[];
  causal_results: InductionEffect[];
  uniform_full_causal_baseline?: number;
  mean_first_repeat_nll?: number;
  mean_repeated_nll?: number;
  forward_attempts: number;
  completed_forwards: number;
  duration_ms: number;
  method_record: Record<string, string>;
  references: { title: string; year: number; url: string }[];
}
