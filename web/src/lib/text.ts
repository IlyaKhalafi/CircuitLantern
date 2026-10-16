export function truncateText(value: string, limit: number) {
  return Array.from(value).slice(0, limit).join("");
}
