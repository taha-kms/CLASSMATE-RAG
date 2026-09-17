import type { AskResponse } from './types';

/**
 * Thin client for the local API (#19).
 *
 * Requests go to a same-origin /api prefix: in development Vite proxies it,
 * and in production FastAPI serves both the bundle and the API, so there is
 * no CORS configuration either way.
 */

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function ask(question: string, course?: string): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, course }),
    });
  } catch {
    // Distinguish "the server is not running" from "the server said no",
    // because on a local-first tool the former is the common case.
    throw new ApiError('Could not reach the local API. Is it running?');
  }

  if (!response.ok) {
    throw new ApiError(`The API returned ${response.status}.`, response.status);
  }

  return (await response.json()) as AskResponse;
}
