import type { ReactNode } from "react";
import { localizedMetadata } from "@/i18n/server";

export async function generateMetadata() {
  return localizedMetadata("Head ablation");
}

export default function Layout({ children }: { children: ReactNode }) {
  return children;
}
