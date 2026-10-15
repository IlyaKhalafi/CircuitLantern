export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    throw new ApiError(
      typeof detail === "string"
        ? detail
        : detail
          ? "Some inputs are invalid. Check your settings."
          : `The server returned ${response.status}. Check that the Python backend is running.`,
      response.status,
    );
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export function post<T>(path: string, body: unknown) {
  return api<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export function bytes(value: number) {
  if (!Number.isFinite(value)) return "Unavailable";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(0)} MiB`;
  return `${(value / 1024).toFixed(0)} KiB`;
}

export function number(value: number | null | undefined, digits = 3) {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return Intl.NumberFormat("en", {
    maximumFractionDigits: digits,
    notation: value > 1e6 ? "compact" : "standard",
  }).format(value);
}

export function downloadJSON(value: unknown, filename: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
