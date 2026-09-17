/**
 * Mirrors what `rag ask` returns, so the UI and the CLI agree on shape.
 * See AskResult and Source in rag/pipeline/rag.py.
 */

export interface Source {
  /** The [n] marker as it appears in the answer text. */
  n: number;
  /** Provenance string for that context block. */
  ref: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  language: string;
  top_k: number;
  hybrid: boolean;
  /** False when the answer cited nothing, so it came from the model (#44). */
  grounded: boolean;
  sources: Source[];
  /** Short line to show muted next to an ungrounded answer. Absent when grounded. */
  notice?: string;
}

export interface ApiError {
  error: string;
}
