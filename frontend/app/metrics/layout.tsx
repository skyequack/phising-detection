import type { Metadata } from "next";

export const metadata: Metadata = { title: "Metrics" };

export default function MetricsLayout({ children }: LayoutProps<"/metrics">) {
  return children;
}
