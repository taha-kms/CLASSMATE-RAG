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

export const SETTING = (over: Record<string, unknown> = {}) => ({
  key: 'K_VECTOR',
  kind: 'int',
  group: 'Retrieval',
  description: 'Candidates fetched from the vector store.',
  default: '8',
  choices: [],
  minimum: 1,
  maximum: 100,
  restart_required: false,
  local_only: false,
  value: '8',
  source: 'file',
  overridden: false,
  file_value: null,
  ...over,
});

export const SETTINGS = [
  SETTING(),
  SETTING({
    key: 'USE_HYBRID',
    kind: 'bool',
    default: 'true',
    value: 'true',
    source: 'default',
    minimum: null,
    maximum: null,
    description: 'Combine keyword and vector results.',
  }),
  SETTING({
    key: 'LLM_PROVIDER',
    kind: 'choice',
    group: 'Answers and models',
    default: 'llama_cpp',
    value: 'llama_cpp',
    choices: ['llama_cpp', 'anthropic', 'openai', 'gemini'],
    minimum: null,
    maximum: null,
    description: 'Which provider generates answers.',
  }),
  SETTING({
    key: 'ENABLE_ROUTING',
    kind: 'bool',
    group: 'Answers and models',
    default: 'false',
    value: 'false',
    source: 'default',
    restart_required: true,
    minimum: null,
    maximum: null,
    description: 'Pick a model per question subject.',
  }),
  SETTING({
    key: 'OPENAI_API_KEY',
    kind: 'secret',
    group: 'Credentials',
    default: '',
    value: '…1234',
    source: 'stored',
    minimum: null,
    maximum: null,
    description: 'Key for the openai provider.',
  }),
  SETTING({
    key: 'OPENAI_BASE_URL',
    kind: 'str',
    group: 'Answers and models',
    default: '',
    value: '',
    source: 'default',
    local_only: true,
    minimum: null,
    maximum: null,
    description: 'Point the openai adapter at another server.',
  }),
];
