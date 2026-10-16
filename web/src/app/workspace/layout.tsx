import type { ReactNode } from "react";
import { localizedMetadata } from "@/i18n/server";

export async function generateMetadata() {
  return localizedMetadata("Model explorer");
}

export default function Layout({ children }: { children: ReactNode }) {
  return children;
}
