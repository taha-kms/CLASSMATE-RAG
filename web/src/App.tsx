import { useState } from 'react';
import { ApiError, ask } from './api';
import type { AskResponse } from './types';

/**
 * A single ask view. The full set of screens (ingest, corpus browser,
 * settings) is #20; this exists so the toolchain has real code to check and
 * the end-to-end tests have a real page to drive.
 */
export default function App() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;

    setPending(true);
    setError(null);
    setResult(null);

    try {
      setResult(await ask(question));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Something went wrong.');
    } finally {
      setPending(false);
    }
  }

  return (
    <main>
      <h1>CLASSMATE-RAG</h1>

      <form onSubmit={onSubmit}>
        <label htmlFor="question">Question</label>
        <input
          id="question"
          name="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="What is the chain rule?"
          autoComplete="off"
        />
        <button type="submit" disabled={pending || !question.trim()}>
          {pending ? 'Asking…' : 'Ask'}
        </button>
      </form>

      {error && <p role="alert">{error}</p>}

      {result && (
        <article aria-label="answer">
          <p>{result.answer}</p>

          {result.notice && <p data-testid="notice">{result.notice}</p>}

          {result.sources.length > 0 && (
            <>
              <h2>Sources</h2>
              <ol>
                {result.sources.map((source) => (
                  <li key={source.n} data-testid="source">
                    [{source.n}] {source.ref}
                  </li>
                ))}
              </ol>
            </>
          )}
        </article>
      )}
    </main>
  );
}
