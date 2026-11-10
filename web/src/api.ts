import { readFrames } from './sse';
import type {
  AskOptions,
  AskResult,
  Chunk,
  ChunkPage,
  DeleteReport,
  Facets,
  Filters,
  IndexStats,
  IngestMeta,
  Job,
  KeyCheck,
  KeyProvider,
  PreviewHit,
  ReconcileReport,
  SettingChange,
  SettingInfo,
  Stage,
} from './types';

/**
 * Client for the local API (docs/api.md).
 *
 * The routes live at the root of the server. In production the bundle is
 * served by that same server, so the base is empty. In development Vite
 * serves the page and proxies /api to the server, stripping the prefix.
 */
const BASE = import.meta.env.DEV ? '/api' : '';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** FastAPI sends `{detail: string}` or, for validation, `{detail: [{msg, loc}]}`. */
function describeDetail(detail: unknown): string | null {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (item && typeof item === 'object' && 'msg' in item ? String(item.msg) : null))
      .filter((m): m is string => m !== null);
    return messages.length > 0 ? messages.join('; ') : null;
  }
  return null;
}

async function failureFrom(response: Response): Promise<ApiError> {
  let detail: string | null = null;
  try {
    detail = describeDetail(((await response.json()) as { detail?: unknown }).detail);
  } catch {
    // Body was not JSON; the status line is all there is.
  }
  return new ApiError(detail ?? `The API returned ${response.status}.`, response.status);
}

async function send(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch (caught) {
    if (caught instanceof DOMException && caught.name === 'AbortError') throw caught;
    // "The server is not running" is the common failure on a local-first
    // tool, and worth telling apart from "the server said no".
    throw new ApiError('Could not reach the local API. Is it running?');
  }
  if (!response.ok) throw await failureFrom(response);
  return response;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  return (await (await send(path, init)).json()) as T;
}

const jsonBody = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

/** Drop empty strings so "no filter" is sent as absence, not as "equals ''". */
function clean<T extends object>(values: T): Partial<T> {
  return Object.fromEntries(
    Object.entries(values).filter(([, v]) => v !== undefined && v !== ''),
  ) as Partial<T>;
}

function query(values: object): string {
  const params = new URLSearchParams(Object.entries(clean(values)).map(([k, v]) => [k, String(v)]));
  const text = params.toString();
  return text ? `?${text}` : '';
}

// ------------------------------------------------------------------ ask

export interface AskHandlers {
  onStage: (stage: Stage) => void;
  onToken: (text: string) => void;
  /** Discard everything streamed so far; what follows supersedes it. */
  onReplace: () => void;
}

/**
 * Stream an answer. Resolves with the `final` event, which is authoritative:
 * citation enforcement and translate-on-miss rewrite the answer after the
 * tokens have gone out.
 */
export async function askStream(
  question: string,
  options: AskOptions,
  handlers: AskHandlers,
  signal?: AbortSignal,
): Promise<AskResult> {
  const response = await send('/ask', {
    ...jsonBody('POST', { question, ...clean(options) }),
    ...(signal ? { signal } : {}),
  });
  if (!response.body) throw new ApiError('The API sent no response body.');

  for await (const frame of readFrames(response.body)) {
    const payload = JSON.parse(frame.data) as Record<string, unknown>;
    switch (frame.event) {
      case 'stage':
        handlers.onStage(payload as unknown as Stage);
        break;
      case 'token':
        handlers.onToken(String(payload.text ?? ''));
        break;
      case 'replace':
        handlers.onReplace();
        break;
      case 'final':
        return payload as unknown as AskResult;
      case 'error':
        throw new ApiError(String(payload.error ?? 'The answer could not be generated.'));
    }
  }
  // A stream that stops without `final` or `error` was cut off.
  throw new ApiError('The connection closed before the answer finished.');
}

// -------------------------------------------------------------- preview

export async function preview(
  question: string,
  options: AskOptions,
): Promise<{ question: string; results: PreviewHit[] }> {
  return json('/preview', jsonBody('POST', { question, ...clean(options) }));
}

// --------------------------------------------------------------- ingest

export async function ingest(file: File, meta: IngestMeta): Promise<Record<string, unknown>> {
  const form = new FormData();
  form.append('file', file);
  // No Content-Type header: the browser must add the multipart boundary.
  return json(`/ingest${query(meta)}`, { method: 'POST', body: form });
}

// --------------------------------------------------------------- corpus

export async function listChunks(
  filters: Filters,
  limit: number,
  offset: number,
): Promise<ChunkPage> {
  return json(`/chunks${query({ ...filters, limit, offset })}`);
}

export async function getChunk(id: string): Promise<Chunk> {
  return json(`/chunks/${encodeURIComponent(id)}`);
}

export async function deleteChunks(
  criteria: Filters & { ids?: string[]; path?: string },
  dryRun: boolean,
): Promise<DeleteReport> {
  return json('/chunks', jsonBody('DELETE', { ...clean(criteria), dry_run: dryRun }));
}

// ------------------------------------------------------------- settings

export const getStats = () => json<IndexStats>('/stats');
export const reconcile = (apply: boolean) =>
  json<ReconcileReport>(`/reconcile${apply ? '?apply=true' : ''}`);
export const vacuum = () => json<Record<string, string>>('/admin/vacuum', { method: 'POST' });

export const startJob = (operation: 'dump' | 'restore' | 'rebuild', body: object) =>
  json<Job>(`/admin/${operation}`, jsonBody('POST', body));

/** Resolves null when the job is unknown: the server restarted and lost it. */
export async function getJob(id: string): Promise<Job | null> {
  try {
    return await json<Job>(`/admin/jobs/${encodeURIComponent(id)}`);
  } catch (caught) {
    if (caught instanceof ApiError && caught.status === 404) return null;
    throw caught;
  }
}

export const getSettings = async () =>
  (await json<{ settings: SettingInfo[] }>('/settings')).settings;

/** `null` removes a setting. All changes are validated first; none apply if one is invalid. */
export const putSettings = async (changes: Record<string, string | null>) =>
  (await json<{ changes: SettingChange[] }>('/settings', jsonBody('PUT', { changes }))).changes;

/** Asks the provider whether the key the server would use is accepted. Sends no key. */
export const validateKey = (provider: KeyProvider) =>
  json<KeyCheck>('/settings/validate', jsonBody('POST', { provider }));

/** The values each metadata field already takes in the corpus. */
export const getFacets = () => json<Facets>('/facets');
