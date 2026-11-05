import { expect, test } from '@playwright/test';
import { json, sse, stream } from './helpers';

// The production bundle (vite preview) calls the API at the root, with no
// /api prefix: in production the same server serves both.

const FINAL = {
  question: 'What is the chain rule?',
  answer:
    'The derivative of a composition multiplies the outer derivative by the inner one [1], applied outermost first [2].',
  language: 'en',
  top_k: 8,
  hybrid: true,
  backend: 'llama_cpp',
  grounded: true,
  sources: [
    { n: 1, ref: '/corpus/calculus-notes.md' },
    { n: 2, ref: '/corpus/slides.pdf' },
  ],
  retrieved: [{ id: 'cm_a' }, { id: 'cm_b' }],
  filters_applied: {},
};

const UNGROUNDED = {
  ...FINAL,
  answer: 'The chain rule is a formula in calculus.',
  grounded: false,
  sources: [],
  retrieved: [],
  notice: "From the model's own knowledge, not your documents.",
};

async function ask(page: import('@playwright/test').Page, question = 'What is the chain rule?') {
  await page.goto('/');
  await page.getByLabel('Question').fill(question);
  await page.getByRole('button', { name: 'Ask' }).click();
}

test('streams an answer, then shows the final text with its sources', async ({ page }) => {
  await page.route('**/ask', (route) =>
    stream(
      route,
      sse(
        ['stage', { stage: 'retrieving', message: 'Searching your documents' }],
        ['token', { text: 'The derivative ' }],
        ['token', { text: 'draft' }],
        ['final', FINAL],
      ),
    ),
  );

  await ask(page);

  await expect(page.getByRole('article', { name: 'answer' })).toContainText('composition');
  // The final event is authoritative over whatever tokens arrived.
  await expect(page.getByRole('article', { name: 'answer' })).not.toContainText('draft');

  const sources = page.getByTestId('source');
  await expect(sources).toHaveCount(2);
  await expect(sources.first()).toContainText('[1] /corpus/calculus-notes.md');
  await expect(sources.nth(1)).toContainText('[2] /corpus/slides.pdf');
});

test('a replace event discards the streamed text', async ({ page }) => {
  await page.route('**/ask', (route) =>
    stream(
      route,
      sse(
        ['token', { text: 'first attempt' }],
        ['replace', { reason: 'translated' }],
        ['token', { text: 'second attempt' }],
        ['final', { ...FINAL, answer: 'second attempt', sources: [], retrieved: [] }],
      ),
    ),
  );

  await ask(page);

  await expect(page.getByRole('article', { name: 'answer' })).toHaveText(/second attempt/);
  await expect(page.getByRole('article', { name: 'answer' })).not.toContainText('first attempt');
});

test('a citation opens the chunk it came from', async ({ page }) => {
  await page.route('**/ask', (route) => stream(route, sse(['final', FINAL])));
  await page.route('**/chunks/cm_a', (route) =>
    json(route, {
      id: 'cm_a',
      text: "The chain rule: (f o g)' = f'(g(x)) g'(x).",
      metadata: { source_path: '/corpus/calculus-notes.md' },
    }),
  );

  await ask(page);
  await page
    .getByRole('article', { name: 'answer' })
    .getByRole('button', { name: '[1]' })
    .first()
    .click();

  await expect(page.getByRole('complementary', { name: 'cited chunk' })).toContainText("f'(g(x))");
});

test('an answer that cites nothing is shown with the muted notice and no sources', async ({
  page,
}) => {
  await page.route('**/ask', (route) => stream(route, sse(['final', UNGROUNDED])));

  await ask(page);

  await expect(page.getByTestId('notice')).toHaveText(
    "From the model's own knowledge, not your documents.",
  );
  await expect(page.getByTestId('source')).toHaveCount(0);
});

test('shows the progress message while the model works', async ({ page }) => {
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  await page.route('**/ask', async (route) => {
    await gate;
    await stream(route, sse(['final', FINAL]));
  });

  await ask(page);
  await expect(page.getByRole('button', { name: 'Asking…' })).toBeDisabled();
  release();
  await expect(page.getByRole('article', { name: 'answer' })).toBeVisible();
});

test('an error event is shown as an error', async ({ page }) => {
  await page.route('**/ask', (route) =>
    stream(
      route,
      sse(
        ['token', { text: 'partial' }],
        ['error', { error: 'The answer could not be generated.' }],
      ),
    ),
  );

  await ask(page);

  await expect(page.getByRole('alert')).toContainText('could not be generated');
});

test('a stream that ends without a final event is reported, not shown as complete', async ({
  page,
}) => {
  await page.route('**/ask', (route) => stream(route, sse(['token', { text: 'cut off' }])));

  await ask(page);

  await expect(page.getByRole('alert')).toContainText('closed before the answer finished');
});

test('says so plainly when the local API is not running', async ({ page }) => {
  await page.route('**/ask', (route) => route.abort('connectionrefused'));

  await ask(page, 'anything');

  await expect(page.getByRole('alert')).toContainText('Could not reach the local API');
});

test('shows the server detail for a rejected request', async ({ page }) => {
  await page.route('**/ask', (route) =>
    json(
      route,
      { detail: [{ loc: ['body', 'question'], msg: 'String should have at least 1 character' }] },
      422,
    ),
  );

  await ask(page, 'anything');

  await expect(page.getByRole('alert')).toContainText('at least 1 character');
});

test('reports a server error distinctly from an unreachable server', async ({ page }) => {
  await page.route('**/ask', (route) => json(route, {}, 500));

  await ask(page, 'anything');

  await expect(page.getByRole('alert')).toContainText('500');
});

test('the ask button stays disabled until there is a question', async ({ page }) => {
  await page.goto('/');

  const button = page.getByRole('button', { name: 'Ask' });
  await expect(button).toBeDisabled();

  await page.getByLabel('Question').fill('   ');
  await expect(button).toBeDisabled();

  await page.getByLabel('Question').fill('a real question');
  await expect(button).toBeEnabled();
});
