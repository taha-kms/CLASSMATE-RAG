import { expect, test } from '@playwright/test';
import type { Route } from '@playwright/test';

const GROUNDED = {
  question: 'What is the chain rule?',
  answer:
    'The derivative of a composition multiplies the outer derivative by the inner one [1], applied outermost first [2].',
  language: 'en',
  top_k: 8,
  hybrid: true,
  grounded: true,
  sources: [
    { n: 1, ref: '/corpus/calculus-notes.md' },
    { n: 2, ref: '/corpus/slides.pdf' },
  ],
};

const UNGROUNDED = {
  ...GROUNDED,
  answer: 'The chain rule is a formula in calculus.',
  grounded: false,
  sources: [],
  notice: "From the model's own knowledge, not your documents.",
};

async function fulfil(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

test('asks a question and shows the answer with its sources', async ({ page }) => {
  await page.route('**/api/ask', (route) => fulfil(route, GROUNDED));

  await page.goto('/');
  await page.getByLabel('Question').fill('What is the chain rule?');
  await page.getByRole('button', { name: 'Ask' }).click();

  await expect(page.getByRole('article', { name: 'answer' })).toContainText('composition');

  // Source numbers must match the [n] markers in the answer text, which is
  // the property #44 introduced the structured Source type to preserve.
  const sources = page.getByTestId('source');
  await expect(sources).toHaveCount(2);
  await expect(sources.first()).toContainText('[1] /corpus/calculus-notes.md');
  await expect(sources.nth(1)).toContainText('[2] /corpus/slides.pdf');
});

test('an answer that cites nothing is shown with the muted notice and no sources', async ({
  page,
}) => {
  await page.route('**/api/ask', (route) => fulfil(route, UNGROUNDED));

  await page.goto('/');
  await page.getByLabel('Question').fill('What is the chain rule?');
  await page.getByRole('button', { name: 'Ask' }).click();

  await expect(page.getByTestId('notice')).toHaveText(
    "From the model's own knowledge, not your documents.",
  );
  await expect(page.getByTestId('source')).toHaveCount(0);
});

test('says so plainly when the local API is not running', async ({ page }) => {
  // The common failure on a local-first tool: the user started the UI but
  // not the backend.
  await page.route('**/api/ask', (route) => route.abort('connectionrefused'));

  await page.goto('/');
  await page.getByLabel('Question').fill('anything');
  await page.getByRole('button', { name: 'Ask' }).click();

  await expect(page.getByRole('alert')).toContainText('Could not reach the local API');
});

test('reports a server error distinctly from an unreachable server', async ({ page }) => {
  await page.route('**/api/ask', (route) => fulfil(route, { error: 'boom' }, 500));

  await page.goto('/');
  await page.getByLabel('Question').fill('anything');
  await page.getByRole('button', { name: 'Ask' }).click();

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
