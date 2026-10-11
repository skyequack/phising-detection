"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Detector" },
  { href: "/history", label: "History" },
  { href: "/model", label: "Model" },
  { href: "/metrics", label: "Metrics" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <nav aria-label="Main" className="border-b border-line bg-panel/60">
      <ul className="mx-auto m-0 flex w-full max-w-5xl list-none gap-1 overflow-x-auto px-4 py-2 sm:px-6">
        {LINKS.map(({ href, label }) => {
          const current = href === "/" ? path === "/" : path.startsWith(href);
          return (
            <li key={href}>
              <Link href={href} aria-current={current ? "page" : undefined}
                className={`block whitespace-nowrap rounded px-3 py-1.5 text-sm transition focus-visible:outline-2 focus-visible:outline-accent ${
                  current ? "bg-accent/15 font-semibold text-accent" : "text-muted hover:text-foreground"}`}>
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
