import { useEffect, useState } from 'react';

/** Retrieval knobs sent with every ask and preview. Kept per browser. */
export interface Knobs {
  top_k: number;
  hybrid: boolean;
}

const KEY = 'classmate-rag.knobs';
const DEFAULTS: Knobs = { top_k: 8, hybrid: true };

function load(): Knobs {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) as Partial<Knobs>;
    const topK = Number(parsed.top_k);
    return {
      top_k: Number.isInteger(topK) && topK >= 1 && topK <= 100 ? topK : DEFAULTS.top_k,
      hybrid: typeof parsed.hybrid === 'boolean' ? parsed.hybrid : DEFAULTS.hybrid,
    };
  } catch {
    // Storage blocked or corrupt: the defaults are always usable.
    return DEFAULTS;
  }
}

export function useKnobs(): [Knobs, (next: Knobs) => void] {
  const [knobs, setKnobs] = useState<Knobs>(load);
  useEffect(() => {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(knobs));
    } catch {
      // Not persisting is fine; the knobs still apply for this session.
    }
  }, [knobs]);
  return [knobs, setKnobs];
}
