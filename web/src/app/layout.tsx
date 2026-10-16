import type { Metadata } from "next";
import { Shell } from "@/components/shell";
import "@fontsource-variable/dm-sans";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Model library · Circuit Lantern",
    template: "%s · Circuit Lantern",
  },
  description:
    "A local mechanistic interpretability workbench for exploring transformer weights, residual streams, attention and predictions.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
