import { expect, test } from '@playwright/test';
import { json } from './helpers';

const CHUNK = (id: string, course = 'Maths') => ({
  id,
  text: `text of ${id}`,
  metadata: { source_path: `/corpus/${id}.md`, course, unit: '3' },
});

test.describe('navigation', () => {
  test('the view is in the hash, so a reload keeps it and never hits an API route', async ({
    page,
  }) => {
    await page.route('**/stats', (route) =>
      json(route, {
        vector_count: 2,
        consistent: true,
        collection: 'c',
        embedding_model: 'e5',
        chroma: { persist_dir: '/i/chroma', disk_bytes: 1048576 },
        bm25: { persist_dir: '/i/bm25', disk_bytes: 2048, count: 2 },
      }),
    );

    await page.goto('/');
    await page.getByRole('link', { name: 'Settings' }).click();
    await expect(page).toHaveURL(/#\/settings$/);

    await page.reload();
    await expect(page.getByRole('region', { name: 'Settings' })).toBeVisible();
    await expect(page.getByTestId('stats')).toContainText('e5');
  });
});

test.describe('preview', () => {
  test('shows fused, vector and BM25 scores per hit', async ({ page }) => {
    await page.route('**/preview', (route) =>
      json(route, {
        question: 'q',
        results: [
          {
            n: 1,
            id: 'cm_a',
            prov: '/corpus/a.md',
            snippet: 'about chains',
            scores: { fused: 0.0328, vector_distance: 0.2232, bm25_score: 0 },
            metadata: {},
          },
        ],
      }),
    );

    await page.goto('/#/preview');
    await page.getByLabel('Query').fill('chain');
    await page.getByRole('button', { name: 'Preview' }).click();

    const row = page.getByTestId('hit');
    await expect(row).toContainText('0.0328');
    await expect(row).toContainText('0.2232');
    await expect(row).toContainText('/corpus/a.md');
  });
});

test.describe('ingest', () => {
  test('uploads the file as multipart with metadata in the query string', async ({ page }) => {
    let url = '';
    let contentType = '';
    await page.route('**/ingest**', (route) => {
      url = route.request().url();
      contentType = route.request().headers()['content-type'] ?? '';
      return json(route, { chunks: 4 }, 201);
    });

    await page.goto('/#/ingest');
    await page.getByLabel('File').setInputFiles({
      name: 'notes.md',
      mimeType: 'text/markdown',
      buffer: Buffer.from('# hello'),
    });
    await page.getByLabel('Course').fill('Maths');
    await page.getByLabel('Unit').fill('3');
    await page.getByRole('button', { name: 'Ingest' }).click();

    await expect(page.getByRole('status')).toContainText('Ingested');
    expect(new URL(url).searchParams.get('course')).toBe('Maths');
    expect(new URL(url).searchParams.get('unit')).toBe('3');
    expect(new URL(url).searchParams.has('author')).toBe(false);
    expect(contentType).toMatch(/^multipart\/form-data; boundary=/);
  });

  test('shows the server reason when the upload is rejected', async ({ page }) => {
    await page.route('**/ingest**', (route) =>
      json(route, { detail: 'The upload needs a filename with an extension.' }, 422),
    );

    await page.goto('/#/ingest');
    await page.getByLabel('File').setInputFiles({
      name: 'README',
      mimeType: 'text/plain',
      buffer: Buffer.from('x'),
    });
    await page.getByRole('button', { name: 'Ingest' }).click();

    await expect(page.getByRole('alert')).toContainText('needs a filename with an extension');
  });
});

test.describe('corpus', () => {
  test('lists chunks and inspects one', async ({ page }) => {
    await page.route('**/chunks?**', (route) =>
      json(route, { count: 2, offset: 0, chunks: [CHUNK('cm_a'), CHUNK('cm_b')] }),
    );

    await page.goto('/#/corpus');
    await expect(page.getByTestId('chunk')).toHaveCount(2);

    await page.getByTestId('chunk').first().getByRole('button', { name: 'Inspect' }).click();
    await expect(page.getByTestId('chunk').first()).toContainText('text of cm_a');
  });

  test('deleting selected chunks shows a dry run and only deletes once confirmed', async ({
    page,
  }) => {
    const requests: Array<Record<string, unknown>> = [];
    await page.route('**/chunks?**', (route) =>
      json(route, { count: 2, offset: 0, chunks: [CHUNK('cm_a'), CHUNK('cm_b')] }),
    );
    await page.route('**/chunks', (route) => {
      if (route.request().method() !== 'DELETE') return route.fallback();
      const body = route.request().postDataJSON() as Record<string, unknown>;
      requests.push(body);
      return body.dry_run
        ? json(route, { dry_run: true, matched: 1, ids: ['cm_a'] })
        : json(route, { dry_run: false, matched: 1, deleted_bm25: 1, deleted_vector: 1 });
    });

    await page.goto('/#/corpus');
    await page.getByLabel('Select cm_a').check();
    await page.getByRole('button', { name: /Delete selected/ }).click();

    await expect(page.getByRole('dialog', { name: 'Confirm delete' })).toContainText('1');
    expect(requests).toEqual([{ ids: ['cm_a'], dry_run: true }]);

    await page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click();
    await expect(page.getByRole('status')).toContainText('Deleted 1 chunk');
    expect(requests[1]).toEqual({ ids: ['cm_a'], dry_run: false });
  });

  test('cannot delete by filter until a filter is set', async ({ page }) => {
    await page.route('**/chunks?**', (route) =>
      json(route, { count: 1, offset: 0, chunks: [CHUNK('cm_a')] }),
    );

    await page.goto('/#/corpus');
    const button = page.getByRole('button', { name: /Delete everything matching/ });
    await expect(button).toBeDisabled();

    await page.getByLabel('Course').fill('Maths');
    await page.getByRole('button', { name: 'Apply filters' }).click();
    await expect(button).toBeEnabled();
  });

  test('pages forward only when the page was full', async ({ page }) => {
    const full = Array.from({ length: 20 }, (_, i) => CHUNK(`cm_${i}`));
    await page.route('**/chunks?**', (route) =>
      json(route, { count: 20, offset: 0, chunks: full }),
    );

    await page.goto('/#/corpus');
    await expect(page.getByRole('button', { name: 'Next' })).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Previous' })).toBeDisabled();
  });
});

test.describe('settings', () => {
  const STATS = {
    vector_count: 5,
    consistent: false,
    collection: 'c',
    embedding_model: 'e5',
    chroma: { persist_dir: '/i/chroma', disk_bytes: 1048576 },
    bm25: { persist_dir: '/i/bm25', disk_bytes: 2048, count: 4 },
  };

  test('repair is offered only after a dry run finds drift', async ({ page }) => {
    const applied: string[] = [];
    await page.route('**/stats', (route) => json(route, STATS));
    await page.route('**/reconcile**', (route) => {
      const apply = new URL(route.request().url()).searchParams.get('apply') === 'true';
      applied.push(apply ? 'apply' : 'dry');
      return json(route, {
        vector_only: apply ? 0 : 1,
        bm25_only: 0,
        removed: apply ? 1 : 0,
        dry_run: !apply,
        sample: ['x'],
      });
    });

    await page.goto('/#/settings');
    await expect(page.getByRole('button', { name: /Remove the vector-only rows/ })).toHaveCount(0);

    await page.getByRole('button', { name: 'Check consistency' }).click();
    await page.getByRole('button', { name: /Remove the vector-only rows/ }).click();

    await expect(page.getByRole('status').first()).toContainText('Removed 1');
    expect(applied).toEqual(['dry', 'apply']);
  });

  test('says so when a job disappears because the server restarted', async ({ page }) => {
    await page.route('**/stats', (route) => json(route, STATS));
    await page.route('**/admin/rebuild', (route) =>
      json(
        route,
        {
          id: 'j1',
          operation: 'rebuild',
          state: 'running',
          started_at: 'now',
          finished_at: null,
          result: null,
          error: null,
        },
        202,
      ),
    );
    await page.route('**/admin/jobs/j1', (route) => json(route, { detail: 'No job' }, 404));

    await page.goto('/#/settings');
    await page.getByLabel('Embedding model').fill('some/model');
    await page.getByRole('button', { name: 'Rebuild embeddings' }).click();

    await expect(page.getByRole('alert')).toContainText('no longer knows this job');
  });

  test('retrieval knobs are sent with a question', async ({ page }) => {
    await page.route('**/stats', (route) => json(route, STATS));
    let body: Record<string, unknown> = {};
    await page.route('**/ask', (route) => {
      body = route.request().postDataJSON() as Record<string, unknown>;
      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'event: error\ndata: {"error":"stop"}\n\n',
      });
    });

    await page.goto('/#/settings');
    await page.getByLabel(/top_k/).fill('3');
    await page.getByLabel(/Hybrid retrieval/).uncheck();

    await page.getByRole('link', { name: 'Chat' }).click();
    await page.getByLabel('Question').fill('q');
    await page.getByRole('button', { name: 'Ask' }).click();

    await expect(page.getByRole('alert')).toBeVisible();
    expect(body).toMatchObject({ question: 'q', top_k: 3, hybrid: false });
  });
});
