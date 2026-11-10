import { expect, test } from '@playwright/test';
import { json, SETTING, SETTINGS } from './helpers';

const CHUNK = (id: string, course = 'Maths') => ({
  id,
  text: `text of ${id}`,
  metadata: { source_path: `/corpus/${id}.md`, course, unit: '3' },
});

test.describe('navigation', () => {
  test('the view is in the hash, so a reload keeps it and never hits an API route', async ({
    page,
  }) => {
    await page.route('**/settings', (route) => json(route, { settings: SETTINGS }));
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

    await page.route('**/facets', (route) => json(route, FACETS));

    await page.goto('/#/ingest');
    await page.getByLabel('File').setInputFiles({
      name: 'notes.md',
      mimeType: 'text/markdown',
      buffer: Buffer.from('# hello'),
    });
    await page.getByLabel('Course').selectOption('Maths');
    // A unit that is not in the list yet: choose "+ Add new" and type it.
    await page.getByLabel('Unit').selectOption({ label: '+ Add new…' });
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

    await page.route('**/facets', (route) => json(route, FACETS));

    await page.goto('/#/corpus');
    const button = page.getByRole('button', { name: /Delete everything matching/ });
    await expect(button).toBeDisabled();

    await page.getByLabel('Course').selectOption('Maths');
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

const FACETS = {
  course: ['Chemistry', 'Maths'],
  unit: ['1', '2', '10'],
  author: [],
};

test.describe('settings', () => {
  const STATS = {
    vector_count: 5,
    consistent: false,
    collection: 'c',
    embedding_model: 'e5',
    chroma: { persist_dir: '/i/chroma', disk_bytes: 1048576 },
    bm25: { persist_dir: '/i/bm25', disk_bytes: 2048, count: 4 },
  };

  test.beforeEach(async ({ page }) => {
    await page.route('**/settings', (route) => {
      if (route.request().method() !== 'GET') return route.fallback();
      return json(route, { settings: SETTINGS });
    });
  });

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

test.describe('server settings', () => {
  let puts: Array<Record<string, unknown>>;

  test.beforeEach(async ({ page }) => {
    puts = [];
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
  });

  test('groups settings and says where each value came from', async ({ page }) => {
    await page.route('**/settings', (route) => json(route, { settings: SETTINGS }));

    await page.goto('/#/settings');

    const row = page.getByTestId('setting-K_VECTOR');
    await expect(row).toContainText('Candidates fetched');
    await expect(row).toContainText('set in .env');
    await expect(page.getByTestId('setting-USE_HYBRID')).toContainText('default');
    await expect(page.getByTestId('setting-ENABLE_ROUTING')).toContainText(
      'Needs a restart to take effect',
    );
  });

  test('sends only the fields that changed and reports what applied', async ({ page }) => {
    await page.route('**/settings', (route) => {
      if (route.request().method() === 'GET') return json(route, { settings: SETTINGS });
      puts.push(route.request().postDataJSON() as Record<string, unknown>);
      return json(route, {
        changes: [
          { key: 'K_VECTOR', action: 'set', applied: true, restart_required: false, note: '' },
          {
            key: 'ENABLE_ROUTING',
            action: 'set',
            applied: false,
            restart_required: true,
            note: 'Takes effect after a restart.',
          },
        ],
      });
    });

    await page.goto('/#/settings');
    const save = page.getByRole('button', { name: 'Save' });
    await expect(save).toBeDisabled();

    await page.locator('summary', { hasText: 'Answers and models' }).click();
    await page.locator('#setting-K_VECTOR').fill('12');
    await page.locator('#setting-ENABLE_ROUTING').check();
    await page.getByRole('button', { name: 'Save 2 changes' }).click();

    await expect(page.getByRole('status').filter({ hasText: 'Saved.' })).toContainText(
      'ENABLE_ROUTING: updated — Takes effect after a restart.',
    );
    await expect(page.getByRole('status').filter({ hasText: 'Saved.' })).toContainText(
      'K_VECTOR: updated — in effect now',
    );
    expect(puts).toEqual([{ changes: { K_VECTOR: '12', ENABLE_ROUTING: 'true' } }]);
  });

  test('a rejected batch shows the reason and keeps what was typed', async ({ page }) => {
    await page.route('**/settings', (route) => {
      if (route.request().method() === 'GET') return json(route, { settings: SETTINGS });
      return json(route, { detail: 'CHUNK_OVERLAP: must be smaller than CHUNK_SIZE (1000).' }, 422);
    });

    await page.goto('/#/settings');
    await page.locator('#setting-K_VECTOR').fill('12');
    await page.getByRole('button', { name: 'Save 1 change' }).click();

    await expect(page.getByRole('alert')).toContainText('must be smaller than CHUNK_SIZE');
    await expect(page.locator('#setting-K_VECTOR')).toHaveValue('12');
  });

  test('a credential field never shows the key and is sent only when typed', async ({ page }) => {
    await page.route('**/settings', (route) => {
      if (route.request().method() === 'GET') return json(route, { settings: SETTINGS });
      puts.push(route.request().postDataJSON() as Record<string, unknown>);
      return json(route, {
        changes: [
          {
            key: 'OPENAI_API_KEY',
            action: 'set',
            applied: true,
            restart_required: false,
            note: '',
          },
        ],
      });
    });

    await page.goto('/#/settings');
    await page.locator('summary', { hasText: 'Credentials' }).click();
    const field = page.locator('#setting-OPENAI_API_KEY');
    await expect(field).toHaveAttribute('type', 'password');
    await expect(field).toHaveValue('');
    await expect(page.getByTestId('setting-OPENAI_API_KEY')).toContainText('Configured (…1234)');

    await field.fill('sk-fake-new-key-0000');
    await page.getByRole('button', { name: 'Save 1 change' }).click();

    await expect(page.getByRole('status').filter({ hasText: 'Saved.' })).toBeVisible();
    expect(puts).toEqual([{ changes: { OPENAI_API_KEY: 'sk-fake-new-key-0000' } }]);
    await expect(page.getByRole('status').filter({ hasText: 'Saved.' })).not.toContainText(
      'sk-fake-new',
    );
    await expect(field).toHaveValue('');
  });

  test('a stored credential can be removed', async ({ page }) => {
    await page.route('**/settings', (route) => {
      if (route.request().method() === 'GET') return json(route, { settings: SETTINGS });
      puts.push(route.request().postDataJSON() as Record<string, unknown>);
      return json(route, {
        changes: [
          {
            key: 'OPENAI_API_KEY',
            action: 'unset',
            applied: true,
            restart_required: false,
            note: '',
          },
        ],
      });
    });

    await page.goto('/#/settings');
    await page.locator('summary', { hasText: 'Credentials' }).click();
    await page
      .getByTestId('setting-OPENAI_API_KEY')
      .getByRole('button', { name: 'Remove it' })
      .click();
    await page.getByRole('button', { name: 'Save 1 change' }).click();

    await expect(page.getByRole('status').filter({ hasText: 'Saved.' })).toContainText('removed');
    expect(puts).toEqual([{ changes: { OPENAI_API_KEY: null } }]);
  });

  test('warns when the shell overrides the value in .env', async ({ page }) => {
    await page.route('**/settings', (route) =>
      json(route, {
        settings: [
          SETTING({ value: '30', source: 'environment', overridden: true, file_value: '8' }),
        ],
      }),
    );

    await page.goto('/#/settings');

    const note = page.getByTestId('setting-K_VECTOR').getByRole('note');
    await expect(note).toContainText('The server is using 30');
    await expect(note).toContainText('.env says 8');
    await expect(note).toContainText('exported in your shell, or a change made outside this page');
  });

  test('says so when the API cannot be reached', async ({ page }) => {
    await page.route('**/settings', (route) => route.abort('connectionrefused'));

    await page.goto('/#/settings');

    await expect(page.getByRole('alert').first()).toContainText('Could not reach the local API');
  });

  test('a setting that decides where data goes is read-only and names the command', async ({
    page,
  }) => {
    await page.route('**/settings', (route) => json(route, { settings: SETTINGS }));

    await page.goto('/#/settings');

    const field = page.locator('#setting-OPENAI_BASE_URL');
    await expect(field).toHaveAttribute('readonly', '');
    await expect(page.getByTestId('setting-OPENAI_BASE_URL')).toContainText(
      'rag config set OPENAI_BASE_URL=VALUE',
    );
  });

  test('a restart-required change is shown next to the value still in use', async ({ page }) => {
    await page.route('**/settings', (route) =>
      json(route, {
        settings: [
          SETTING({
            key: 'LOG_LEVEL',
            kind: 'choice',
            choices: ['INFO', 'DEBUG'],
            default: 'INFO',
            value: 'INFO',
            restart_required: true,
            pending_value: 'DEBUG',
            minimum: null,
            maximum: null,
          }),
        ],
      }),
    );

    await page.goto('/#/settings');

    const note = page.getByTestId('pending-LOG_LEVEL');
    await expect(note).toContainText('Saved: DEBUG');
    await expect(note).toContainText('still using INFO');
    await expect(note).toContainText('after you restart it');
  });
  test.describe('testing a saved key', () => {
    const check = (over: Record<string, unknown>) => ({
      provider: 'openai',
      outcome: 'valid',
      ok: true,
      message: 'The provider accepted the key.',
      source: 'stored',
      masked: '********1234',
      target: 'https://api.openai.com/v1/models',
      status: 200,
      ...over,
    });

    test('asks the server to test the provider, sending no key, and shows which key was used', async ({
      page,
    }) => {
      const posts: unknown[] = [];
      await page.route('**/settings', (route) => json(route, { settings: SETTINGS }));
      await page.route('**/settings/validate', (route) => {
        posts.push(route.request().postDataJSON());
        return json(route, check({ source: 'environment', masked: '********9999' }));
      });

      await page.goto('/#/settings');
      await page.locator('summary', { hasText: 'Credentials' }).click();
      await page
        .getByTestId('setting-OPENAI_API_KEY')
        .getByRole('button', { name: 'Test key' })
        .click();

      const row = page.getByTestId('setting-OPENAI_API_KEY');
      await expect(row).toContainText('Accepted.');
      await expect(row).toContainText('key set in the server’s environment');
      await expect(row).toContainText('9999');
      await expect(row).toContainText('api.openai.com');
      expect(posts).toEqual([{ provider: 'openai' }]);
    });

    test('says plainly when the key is not accepted', async ({ page }) => {
      await page.route('**/settings', (route) => json(route, { settings: SETTINGS }));
      await page.route('**/settings/validate', (route) =>
        json(
          route,
          check({
            outcome: 'rejected',
            ok: false,
            status: 401,
            message: 'The provider rejected the key.',
          }),
        ),
      );

      await page.goto('/#/settings');
      await page.locator('summary', { hasText: 'Credentials' }).click();
      const row = page.getByTestId('setting-OPENAI_API_KEY');
      await row.getByRole('button', { name: 'Test key' }).click();

      await expect(row).toContainText('Not accepted.');
      await expect(row).toContainText('rejected the key');
    });

    test('is unavailable while a new key is typed, because it tests the saved one', async ({
      page,
    }) => {
      await page.route('**/settings', (route) => json(route, { settings: SETTINGS }));
      await page.goto('/#/settings');
      await page.locator('summary', { hasText: 'Credentials' }).click();

      await page.locator('#setting-OPENAI_API_KEY').fill('sk-fake-new-key-0000');

      const row = page.getByTestId('setting-OPENAI_API_KEY');
      await expect(row.getByRole('button', { name: 'Test key' })).toBeDisabled();
      await expect(row).toContainText('save the new key first');
    });

    test('shows a failed request instead of nothing', async ({ page }) => {
      await page.route('**/settings', (route) => json(route, { settings: SETTINGS }));
      await page.route('**/settings/validate', (route) =>
        route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' }),
      );
      await page.goto('/#/settings');
      await page.locator('summary', { hasText: 'Credentials' }).click();
      const row = page.getByTestId('setting-OPENAI_API_KEY');
      await row.getByRole('button', { name: 'Test key' }).click();

      await expect(row.getByRole('status')).not.toBeEmpty();
    });
  });
});

test.describe('dropdown fields', () => {
  test('a filter offers what is already in the corpus, and nothing to add', async ({ page }) => {
    await page.route('**/facets', (route) => json(route, FACETS));

    await page.goto('/#/preview');
    const course = page.getByLabel('Course');

    await expect(course.locator('option')).toHaveText(['Any', 'Chemistry', 'Maths']);
    await expect(page.getByLabel('Unit').locator('option')).toHaveText(['Any', '1', '2', '10']);
    // Asking for a course that does not exist can only return nothing.
    await expect(course.locator('option', { hasText: 'Add new' })).toHaveCount(0);
  });

  test('language is a fixed list, and type and semester are not offered', async ({ page }) => {
    await page.route('**/facets', (route) => json(route, FACETS));

    await page.goto('/#/preview');
    await expect(page.getByLabel('Language').locator('option')).toHaveText(['Any', 'en', 'it']);
    await expect(page.getByLabel('Type')).toHaveCount(0);
    await expect(page.getByLabel('Semester')).toHaveCount(0);
  });

  test('ingesting can add a new course, and go back to the list', async ({ page }) => {
    await page.route('**/facets', (route) => json(route, FACETS));

    await page.goto('/#/ingest');
    const course = page.getByLabel('Course');
    await expect(course.locator('option').last()).toHaveText('+ Add new…');
    // Language still cannot gain values: the server only accepts these.
    await expect(page.getByLabel('Language').locator('option')).toHaveText([
      'Not set',
      'en',
      'it',
      'auto',
    ]);

    await course.selectOption({ label: '+ Add new…' });
    await page.getByLabel('Course').fill('Physics');
    await expect(page.getByLabel('Course')).toHaveValue('Physics');

    await page.getByRole('button', { name: 'Back to list' }).click();
    await expect(page.getByLabel('Course')).toHaveValue('');
    await expect(page.getByLabel('Course').locator('option')).toHaveCount(4);
  });

  test('the fields still work when the server cannot list values', async ({ page }) => {
    await page.route('**/facets', (route) => route.fulfill({ status: 500, body: 'no' }));

    await page.goto('/#/preview');
    await expect(page.getByLabel('Course').locator('option')).toHaveText(['Any']);
    await expect(page.getByLabel('Language').locator('option')).toHaveText(['Any', 'en', 'it']);
  });
});
