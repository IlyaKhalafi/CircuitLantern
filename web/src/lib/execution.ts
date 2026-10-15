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

export function executionLabel(precision?: string) {
  return (
    executionFormats.find(([value]) => value === precision)?.[1] ??
    precision ??
    "FP32"
  );
}
