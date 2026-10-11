import { useEffect, useRef, useState } from "react";

type State<T> = { key: string; data: T | null; error: string | null };

/** Fetch on mount and whenever `key` changes. Keeps the previous data visible while the next request is in flight. */
export function useFetch<T>(fetcher: () => Promise<T>, key = "") {
  const latest = useRef(fetcher);
  const [state, setState] = useState<State<T> | null>(null);

  useEffect(() => {
    latest.current = fetcher;
  });

  useEffect(() => {
    let live = true;
    latest
      .current()
      .then((data) => live && setState({ key, data, error: null }))
      .catch((e) => live && setState({ key, data: null, error: e instanceof Error ? e.message : "Something went wrong." }));
    return () => {
      live = false;
    };
  }, [key]);

  const loading = state?.key !== key;
  return { data: state?.data ?? null, error: loading ? null : state?.error ?? null, loading };
}
