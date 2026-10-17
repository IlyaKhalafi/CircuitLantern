import type { ReactNode } from "react";
import { localizedMetadata } from "@/i18n/server";

export async function generateMetadata() {
  const { title } = await localizedMetadata("Residual patching");
  return { title: { default: title, template: "%s · Circuit Lantern" } };
}

export default function Layout({ children }: { children: ReactNode }) {
  return children;
}
