import type { Metadata } from "next";

export const metadata: Metadata = { title: "History" };

export default function HistoryLayout({ children }: LayoutProps<"/history">) {
  return children;
}
