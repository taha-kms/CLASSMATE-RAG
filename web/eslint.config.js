import js from '@eslint/js';
import prettier from 'eslint-config-prettier';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'playwright-report', 'test-results', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  // .configs.flat.* is the flat-config form; the top-level keys are
  // still eslintrc-shaped and ESLint 9 rejects them.
  reactHooks.configs.flat['recommended-latest'],
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
  },
  // Formatting is Prettier's job. Keeping ESLint out of it avoids the two
  // disagreeing, the same way ruff lint and ruff format are kept separate
  // on the Python side.
  prettier,
);
