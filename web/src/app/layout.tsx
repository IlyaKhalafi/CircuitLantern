import type { Metadata } from "next";
import { Shell } from "@/components/shell";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Circuit Lantern · Model library",
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
