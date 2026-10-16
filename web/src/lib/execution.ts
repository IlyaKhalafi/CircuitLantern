export const executionFormats = [
  ["auto", "Auto detect"],
  ["float32", "FP32"],
  ["float16", "FP16"],
  ["bfloat16", "BF16"],
  ["int4", "NF4 · 4 bit"],
  ["int8", "LLM.int8 · 8 bit"],
  ["fp8", "FP8 · runtime pending"],
  ["nvfp4", "NVFP4 · runtime pending"],
] as const;

export function fourBitFormat(quantization: unknown): "fp4" | "nf4" | null {
  if (
    !quantization ||
    typeof quantization !== "object" ||
    Array.isArray(quantization)
  )
    return null;
  const config = quantization as Record<string, unknown>;
  if (config.load_in_4bit !== true && config._load_in_4bit !== true)
    return null;
  if (config.quant_method && config.quant_method !== "bitsandbytes")
    return null;
  const format =
    config.bnb_4bit_quant_type === undefined
      ? "fp4"
      : config.bnb_4bit_quant_type;
  return format === "fp4" || format === "nf4" ? format : null;
}

export function executionLabel(
  precision?: string,
  configuration?: Record<string, unknown> | null,
) {
  const resolved =
    precision ??
    (typeof configuration?.precision === "string"
      ? configuration.precision
      : undefined);
  if (resolved === "int4") {
    const format = fourBitFormat(configuration?.quantization);
    if (format === "fp4") return "FP4 · bitsandbytes";
    if (format === "nf4") return "NF4 · 4 bit";
    return "4 bit · bitsandbytes";
  }
  return (
    executionFormats.find(([value]) => value === resolved)?.[1] ??
    resolved ??
    "FP32"
  );
}
