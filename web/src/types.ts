/**
 * Shapes of the local API (docs/api.md, rag/api/). Server-side dicts that are
 * only passed through are typed loosely on purpose: a field the UI does not
 * read is not worth pinning.
 */

export interface Source {
  /** The [n] marker as it appears in the answer text. */
  n: number;
  /** Provenance string for that context block. */
  ref: string;
}

/** The `final` event of /ask: the full AskResult. */
export interface AskResult {
  question: string;
  answer: string;
  language: string;
  top_k: number;
  hybrid: boolean;
  backend: string;
  /** False when the answer cited nothing, so it came from the model. */
  grounded: boolean;
  sources: Source[];
  retrieved: Array<{ id?: string; [key: string]: unknown }>;
  filters_applied: Record<string, unknown>;
  /** Short line to show muted next to an ungrounded answer. */
  notice?: string | null;
  route?: string | null;
  route_reason?: string | null;
}

export interface Stage {
  stage: string;
  message: string;
}

export interface Filters {
  course?: string;
  unit?: string;
  language?: string;
  doc_type?: string;
  author?: string;
  semester?: string;
}

export interface AskOptions extends Filters {
  top_k: number;
  hybrid: boolean;
}

export interface PreviewHit {
  n: number;
  id: string;
  prov: string;
  snippet: string;
  scores: { fused?: number; vector_distance?: number | null; bm25_score?: number | null };
  metadata: Record<string, unknown>;
}

export interface Chunk {
  id: string;
  text: string;
  metadata: Record<string, unknown>;
}

export interface ChunkPage {
  count: number;
  offset: number;
  chunks: Chunk[];
}

export interface DeleteReport {
  dry_run: boolean;
  matched: number;
  ids?: string[];
  deleted_bm25?: number;
  deleted_vector?: number;
}

export interface IndexStats {
  vector_count: number;
  consistent: boolean;
  collection: string;
  embedding_model: string;
  chroma: { persist_dir: string; disk_bytes: number };
  bm25: { persist_dir: string; disk_bytes: number; count: number };
}

export interface ReconcileReport {
  vector_only: number;
  bm25_only: number;
  removed: number;
  dry_run: boolean;
  sample: string[];
}

export interface Job {
  id: string;
  operation: string;
  state: 'running' | 'succeeded' | 'failed';
  started_at: string;
  finished_at: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface IngestMeta {
  course?: string;
  unit?: string;
  language?: string;
  doc_type?: string;
  author?: string;
  semester?: string;
  tags?: string;
}

export type SettingKind = 'str' | 'int' | 'float' | 'bool' | 'choice' | 'secret';

export interface SettingInfo {
  key: string;
  kind: SettingKind;
  group: string;
  description: string;
  default: string;
  choices: string[];
  minimum: number | null;
  maximum: number | null;
  restart_required: boolean;
  /** Can only be changed with `rag config set` on this machine, not over HTTP. */
  local_only: boolean;
  /** Effective value; masked for credentials, empty when unset. */
  value: string;
  source: 'environment' | 'file' | 'default' | 'stored' | 'unset';
  /** The shell holds a different value than .env, so saving to the file changes nothing. */
  overridden: boolean;
  file_value: string | null;
  /** Saved to .env but waiting for a restart; `value` is still what the server is using. */
  pending_value: string | null;
}

export interface SettingChange {
  key: string;
  action: 'set' | 'unset';
  applied: boolean;
  restart_required: boolean;
  note: string;
}

export type KeyProvider = 'anthropic' | 'openai' | 'gemini';

/** The provider's answer to "is this key accepted?". Never contains the key. */
export interface KeyCheck {
  provider: KeyProvider;
  outcome: 'valid' | 'rejected' | 'rate_limited' | 'unreachable' | 'unexpected' | 'missing';
  ok: boolean;
  message: string;
  /** Which key was tested: the server's environment wins over the stored one. */
  source: 'environment' | 'stored' | '';
  masked: string;
  /** Where the request went. A local OpenAI-compatible server may accept any key. */
  target: string;
  status: number | null;
}

/** Values already in the corpus for each metadata field, for dropdowns. */
export type Facets = Record<'course' | 'unit' | 'author' | 'semester', string[]>;
