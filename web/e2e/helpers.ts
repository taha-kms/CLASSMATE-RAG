import type { Route } from '@playwright/test';

export async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

/** Build a Server-Sent Events body the way rag/api/app.py frames it. */
export function sse(...events: Array<[string, unknown]>): string {
  return events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join('');
}

export async function stream(route: Route, body: string) {
  await route.fulfill({ status: 200, contentType: 'text/event-stream', body });
}
