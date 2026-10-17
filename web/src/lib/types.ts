import type { PromptSetResult } from "@/lib/prompt-sets";
import type { InductionResult } from "@/lib/induction";

export interface ExecutionProvenance {
  execution_id?: string | null;
  execution_precision?: string;
  execution_context_length?: number;
  execution_config?: Record<string, unknown> | null;
}
export interface ModelCard {
  downloads?: number | null;
  likes?: number | null;
  updated_at?: string | null;
  created_at?: string | null;
  id: string;
  name: string;
  family: string;
  parameters: string;
  description: string;
  tag: string;
  color: string;
}
export interface ModelMetadata extends ExecutionProvenance {
  id: string;
  revision: string;
  architecture: string;
  parameters: number;
  parameter_estimated: boolean;
  weight_bytes: number;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
  num_kv_heads: number;
  context_length: number;
  vocab_size: number;
  dtype: string;
  quantization: Record<string, unknown> | null;
  tied_embeddings?: boolean;
  quantizable_parameters?: number;
  largest_matrix_elements?: number;
  model_buffer_bytes?: number;
  largest_model_buffer_bytes?: number;
  download_bytes?: number;
}
export interface GPU {
  index: number;
  name: string;
  total_bytes: number;
  free_bytes: number;
  compute_capability: number;
  driver: string;
}
export interface Hardware {
  cpu: number;
  cpu_threads: number;
  ram_total_bytes: number;
  ram_available_bytes: number;
  ram_scope?: "system" | "cgroup";
  ram_host_total_bytes?: number;
  ram_host_available_bytes?: number;
  ram_constraints?: {
    source: string;
    limit_bytes: number;
    used_bytes: number | null;
    reclaimable_bytes: number;
  }[];
  disk_total_bytes: number;
  disk_free_bytes: number;
  gpus: GPU[];
  packages: Record<string, string | null>;
  cuda_available: boolean;
  runtime_device: string;
  execution_gpu?: Omit<GPU, "driver"> | null;
  cuda_error?: string | null;
  quantization?: {
    version: string | null;
    cpu: Record<string, { supported: boolean; reason?: string }>;
    cuda: Record<string, { supported: boolean; reason?: string }>;
  };
}
export interface Compatibility {
  reclaimable_bytes?: number;
  source_bytes?: number;
  cached_bytes?: number;
  context_length?: number;
  compatible: boolean;
  status: string;
  precision: string;
  runtime_precision: string;
  device: string;
  weight_bytes: number;
  analysis_weight_bytes: number;
  analysis_workspace_bytes?: number;
  model_buffer_bytes?: number;
  model_buffer_workspace_bytes?: number;
  workspace_bytes?: number;
  activation_bytes?: number;
  logits_bytes?: number;
  kv_cache_bytes: number;
  overhead_bytes: number;
  required_bytes: number;
  available_bytes: number;
  download_bytes: number;
  blockers: string[];
  warnings: string[];
  metadata: ModelMetadata;
}
export interface Job {
  cancel_requested?: boolean;
  id: string;
  model_id?: string;
  requires_model?: boolean;
  status: string;
  progress: number;
  message: string;
}
export interface AttentionHeadTarget {
  layer: number;
  heads: number;
  head_dim: number;
}
export interface HeadAblation {
  layer: number;
  head: number;
  heads?: number;
  head_dim?: number;
  projection?: string;
  scope?: string;
  method?: string;
}
export interface RuntimeState {
  attention_head_targets?: AttentionHeadTarget[];
  context_length?: number;
  max_prompt_tokens?: number;
  max_head_sweep_heads?: number;
  precision?: string;
  loaded: boolean;
  metadata: ModelMetadata | null;
  device: string;
}
export interface Weight {
  name: string;
  shape: number[];
  elements: number;
  dtype: string;
}
export interface BlockFlow {
  layer: number;
  supported: boolean;
  parallel_residual?: boolean;
  provenance?: "source";
  block_type?: string;
  mixer_type?: string;
  mapped_matrices?: number;
  total_matrices?: number;
  opaque_modules?: string[];
  unmapped_matrices?: string[];
  assumptions?: string[];
  nodes: {
    id: string;
    label: string;
    x: number;
    y: number;
    weight: string | null;
    detail: string;
    section?: string;
    module_path?: string;
    matrices?: string[];
    opaque?: boolean;
    boundary?: "input" | "output";
    boundary_origin?: string;
  }[];
  edges: {
    source: string;
    target: string;
    label: string;
    residual: boolean;
    branch?: boolean;
  }[];
  scope?: string;
}
export interface ModelGraph {
  dataflow?: BlockFlow[];
  metadata: ModelMetadata;
  matrices: Weight[];
  total_parameters: number;
}
export interface MatrixMetrics extends ExecutionProvenance {
  analysis_id?: string;
  device?: string;
  model_id?: string;
  revision?: string;
  operator_1_norm?: number;
  nuclear_norm?: number | null;
  effective_rank?: number | null;
  numerical_rank?: number | null;
  rank_tolerance?: number | null;
  spectrum_precision?: string;
  source_format?: string;
  quantized?: boolean;
  analysis_note?: string;
  name: string;
  shape: number[];
  elements: number;
  dtype: string;
  frobenius_norm: number;
  l1_norm: number;
  infinity_norm: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  sparsity: number;
  spectral_norm: number;
  spectral_radius: number | null;
  stable_rank: number;
  condition_number: number | null;
  singular_values: number[];
  cumulative_energy: number[];
  spectrum_exact: boolean;
  spectrum_shape: number[];
  spectrum_method: string;
  histogram: { value: number; count: number }[];
  histogram_sampled: boolean;
  heatmap: number[][];
  heatmap_method: string;
}
export interface Prediction {
  token: string;
  probability: number;
  token_id: number;
  logit?: number;
}
export interface LayerTrace {
  token_target_log_probabilities?: (number | null)[][];
  token_target_ranks?: (number | null)[][];
  token_prediction_ids?: number[];
  token_prediction_probabilities?: number[];
  token_entropies?: number[];
  update_norm?: number | null;
  update_rms?: number | null;
  relative_update?: number | null;
  layer: number;
  label: string;
  rms: number;
  norm: number;
  token_norms: number[];
  cosine_similarity: number | null;
  entropy: number;
  predictions: Prediction[];
  token_predictions: string[];
}
export interface Attention {
  query_position?: number;
  query_by_head?: number[][];
  query_head_entropy?: number[];
  layer: number;
  pattern: number[][];
  head_entropy: number[];
  heads: number;
}
export interface TraceGeometry {
  labels: string[];
  cka: (number | null)[][];
  token_cosine: (number | null)[][];
  sampled_positions: number[];
  token_count: number;
  minimum_samples: number;
  undefined_layers: number[];
  method: string;
}
export interface TargetTokenPreview {
  requested_text: string;
  token_count: number;
  token_ids: number[];
  tokens: string[];
  pieces: (string | null)[];
  special_tokens: boolean[];
  truncated: boolean;
  skipped: boolean;
  valid: boolean;
  error: string | null;
  duplicate_of: number | null;
}
export interface TargetPreviewResponse extends ExecutionProvenance {
  model_id: string;
  revision: string;
  targets: TargetTokenPreview[];
  valid: boolean;
  max_targets: number;
  max_target_characters: number;
}
export interface TrackedTarget {
  requested_text: string;
  token_id: number;
  token: string;
}
export interface Trace extends ExecutionProvenance {
  tracked_targets?: TrackedTarget[];
  target_score_method?: string;
  target_rank_method?: string;
  token_prediction_method?: string;
  final_state_normalized?: boolean;
  geometry?: TraceGeometry;
  model_id: string;
  revision: string;
  prompt: string;
  tokens: string[];
  token_ids: number[];
  continuation: string;
  decoded_prompt?: string;
  generated_text?: string;
  continuation_text?: string | null;
  generated_token_ids?: number[];
  generated_tokens?: string[];
  generated_pieces?: string[];
  capture_attention?: boolean;
  attention_query_position?: number | null;
  generation_performed?: boolean;
  requested_max_new_tokens?: number;
  generation_settings?: null | {
    max_new_tokens: number;
    do_sample: boolean;
    token_healing: boolean;
    num_beams: number;
    num_return_sequences: number;
    use_cache: boolean;
    pad_token_id: number | null;
    return_dict_in_generate: boolean;
    output_attentions: boolean;
    output_hidden_states: boolean;
    output_scores: boolean;
    output_logits: boolean;
  };
  layers: LayerTrace[];
  attention: Attention[];
  predictions: Prediction[];
  duration_ms: number;
  device: string;
  ablate_layer: number | null;
  ablate_head?: HeadAblation | null;
  method: string;
}
interface ExperimentBase {
  source_analysis_id?: string;
  source_job_id?: string | null;
  id: string;
  name: string;
  notes: string;
  created_at: string;
  model_id: string;
}
export type Experiment =
  | (ExperimentBase & { kind?: "trace"; trace?: Trace })
  | (ExperimentBase & { kind: "causal"; result?: CausalResult })
  | (ExperimentBase & { kind: "causal_batch"; result?: PromptSetResult })
  | (ExperimentBase & { kind: "head_sweep"; result?: HeadSweepResult })
  | (ExperimentBase & { kind: "matrix"; result?: MatrixMetrics })
  | (ExperimentBase & { kind: "induction_probe"; result?: InductionResult });

export interface CausalLayer {
  layer: number;
  probability: number;
  log_probability: number;
  delta_log_probability: number;
  recovery: number | null;
}
export interface CausalResult extends ExecutionProvenance {
  error?: string | null;
  failed_layer?: number | null;
  device?: string;
  complete?: boolean;
  requested_target: string;
  total_layers: number;
  cancelled: boolean;
  baseline_log_probability_gap: number;
  recovery_minimum_gap: number;
  recovery_warning: string | null;
  model_id: string;
  revision: string;
  clean_prompt: string;
  corrupted_prompt: string;
  tokens: string[];
  token_ids?: number[];
  corrupted_token_ids?: number[];
  corrupted_tokens: string[];
  target_token: string;
  target_token_id: number;
  position: number | null;
  clean: { probability: number; log_probability: number };
  corrupted: { probability: number; log_probability: number };
  layers: CausalLayer[];
  duration_ms: number;
  recovery_defined: boolean;
  method: string;
}

export interface TokenizedPrompt {
  token_count: number;
  token_ids: number[];
  tokens: string[];
  pieces: string[];
  special_tokens: boolean[];
  truncated: boolean;
  fits: boolean;
}
export interface TokenPreviewResponse extends ExecutionProvenance {
  model_id: string;
  revision: string;
  token_limit: number;
  reserved_tokens: number;
  prompts: TokenizedPrompt[];
}

export interface HeadSweepScore {
  log_probability: number;
  probability: number;
}
export interface HeadSweepRow extends HeadSweepScore {
  layer: number;
  head: number;
  projection: string;
  delta_log_probability: number;
  kl_divergence: number;
  top_token_id: number;
  top_token: string;
}
export interface HeadSweepResult extends ExecutionProvenance {
  complete: boolean;
  error: string | null;
  failed_head: { layer: number; head: number } | null;
  logits_dtype: string;
  last_token_logits_only: boolean;
  model_id: string;
  revision: string;
  device: string;
  prompt: string;
  tokens: string[];
  token_ids: number[];
  requested_target: string;
  target_token_id: number;
  target_token: string;
  baseline: HeadSweepScore;
  baseline_top_token_id: number;
  layer_targets: AttentionHeadTarget[];
  heads: HeadSweepRow[];
  total_heads: number;
  duration_ms: number;
  cancelled: boolean;
  scope: string;
  method: string;
}
