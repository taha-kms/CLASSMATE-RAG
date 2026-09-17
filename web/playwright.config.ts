import { defineConfig, devices } from '@playwright/test';

/**
 * The API (#19) does not exist yet, so these drive the built bundle and
 * fulfil /api/* from the tests themselves. That is not a stopgap: it keeps
 * the UI's behaviour testable independently of whether generation is fast,
 * or a model is present, or Chroma is up. The flows that genuinely need a
 * live backend are listed in the issue and land with #19 and #20.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  // A test that only passes on a retry is a flaky test, and CI should say so.
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',

  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'on-first-retry',
  },

  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],

  // Preview serves the production build, so these exercise what actually
  // ships rather than the dev server.
  webServer: {
    // --host 127.0.0.1 is load-bearing. vite preview binds 'localhost',
    // which resolves to ::1 on the CI runner while Playwright polls the
    // IPv4 address, so the server never appears to come up.
    command: 'npm run preview -- --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
