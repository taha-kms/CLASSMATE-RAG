import { useState } from 'react';
import { ApiError, preview } from './api';
import { FilterFields } from './FilterFields';
import type { Knobs } from './settings-store';
import type { Filters, PreviewHit } from './types';

const fixed = (value: number | null | undefined, digits = 4) =>
  typeof value === 'number' ? value.toFixed(digits) : '—';

/** What the retriever returned, with the scores that produced the ranking. */
export default function Preview({ knobs }: { knobs: Knobs }) {
  const [question, setQuestion] = useState('');
  const [filters, setFilters] = useState<Filters>({});
  const [hits, setHits] = useState<PreviewHit[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setPending(true);
    setError(null);
    try {
      setHits((await preview(question, { ...filters, ...knobs })).results);
    } catch (caught) {
      setHits(null);
      setError(caught instanceof ApiError ? caught.message : 'Something went wrong.');
    } finally {
      setPending(false);
    }
  }

  return (
    <section aria-label="Preview">
      <form onSubmit={onSubmit}>
        <label htmlFor="preview-question">Query</label>
        <input
          id="preview-question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          autoComplete="off"
        />
        <details>
          <summary>Filters</summary>
          <FilterFields value={filters} onChange={setFilters} />
        </details>
        <button type="submit" disabled={pending || !question.trim()}>
          {pending ? 'Searching…' : 'Preview'}
        </button>
      </form>

      {error && <p role="alert">{error}</p>}
      {hits && hits.length === 0 && <p>Nothing matched.</p>}

      {hits && hits.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Fused</th>
                <th>Vector distance</th>
                <th>BM25</th>
                <th>Source</th>
                <th>Snippet</th>
              </tr>
            </thead>
            <tbody>
              {hits.map((hit) => (
                <tr key={hit.id} data-testid="hit">
                  <td>{hit.n}</td>
                  <td>{fixed(hit.scores.fused)}</td>
                  <td>{fixed(hit.scores.vector_distance)}</td>
                  <td>{fixed(hit.scores.bm25_score)}</td>
                  <td>{hit.prov}</td>
                  <td>{hit.snippet}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
