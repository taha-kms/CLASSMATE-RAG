import { useEffect, useState } from 'react';
import { ApiError, deleteChunks, listChunks } from './api';
import { FilterFields, hasFilter } from './FilterFields';
import type { Chunk, DeleteReport, Filters } from './types';

const PAGE_SIZE = 20;

interface Plan {
  criteria: Filters & { ids?: string[] };
  report: DeleteReport;
}

export default function Corpus() {
  const [filters, setFilters] = useState<Filters>({});
  const [applied, setApplied] = useState<Filters>({});
  const [offset, setOffset] = useState(0);
  const [reloads, setReloads] = useState(0);
  const [loaded, setLoaded] = useState<{
    key: string;
    chunks: Chunk[];
    error: string | null;
  } | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [inspected, setInspected] = useState<string | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Whatever the current filters, page and reload count are, that is the key
  // the displayed rows must have been fetched for; otherwise we are loading.
  const key = JSON.stringify([applied, offset, reloads]);

  useEffect(() => {
    let cancelled = false;
    listChunks(applied, PAGE_SIZE, offset)
      .then((page) => {
        if (!cancelled) setLoaded({ key, chunks: page.chunks, error: null });
      })
      .catch((caught: unknown) => {
        if (cancelled) return;
        const message = caught instanceof ApiError ? caught.message : 'Something went wrong.';
        setLoaded({ key, chunks: [], error: message });
      });
    return () => {
      cancelled = true;
    };
  }, [key, applied, offset]);

  const loading = loaded?.key !== key;
  const chunks = loaded && !loading ? loaded.chunks : [];
  const error = actionError ?? (loaded && !loading ? loaded.error : null);

  // `count` from the API is the size of this page, not the corpus, so a full
  // page is the only signal that another one may follow.
  const hasNext = chunks.length === PAGE_SIZE;

  function toggle(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  /** Every delete is a dry run first; nothing is removed until it is confirmed. */
  async function propose(criteria: Plan['criteria']) {
    setActionError(null);
    setNotice(null);
    try {
      setPlan({ criteria, report: await deleteChunks(criteria, true) });
    } catch (caught) {
      setActionError(caught instanceof ApiError ? caught.message : 'Something went wrong.');
    }
  }

  async function confirm() {
    if (!plan) return;
    try {
      const done = await deleteChunks(plan.criteria, false);
      setNotice(
        `Deleted ${done.matched} chunk${done.matched === 1 ? '' : 's'} ` +
          `(${done.deleted_bm25 ?? 0} keyword, ${done.deleted_vector ?? 0} vector).`,
      );
      setPlan(null);
      setSelected(new Set());
      setReloads((count) => count + 1);
    } catch (caught) {
      setActionError(caught instanceof ApiError ? caught.message : 'Something went wrong.');
    }
  }

  return (
    <section aria-label="Corpus">
      <form
        onSubmit={(event) => {
          event.preventDefault();
          setOffset(0);
          setApplied(filters);
          setSelected(new Set());
          setActionError(null);
        }}
      >
        <FilterFields value={filters} onChange={setFilters} />
        <button type="submit">Apply filters</button>
      </form>

      {error && <p role="alert">{error}</p>}
      {notice && <p role="status">{notice}</p>}
      {loading && <p className="muted">Loading…</p>}
      {!loading && !error && chunks.length === 0 && <p>No chunks.</p>}

      {chunks.length > 0 && (
        <>
          <div className="row">
            <button
              type="button"
              disabled={selected.size === 0}
              onClick={() => propose({ ids: [...selected] })}
            >
              Delete selected ({selected.size})
            </button>
            {/* An empty filter set would match the whole corpus, so this
                button does not exist until there is something to narrow by. */}
            <button
              type="button"
              disabled={!hasFilter(applied)}
              title={hasFilter(applied) ? undefined : 'Set a filter first'}
              onClick={() => propose(applied)}
            >
              Delete everything matching these filters
            </button>
          </div>

          <table>
            <thead>
              <tr>
                <th />
                <th>Source</th>
                <th>Course</th>
                <th>Unit</th>
                <th>Text</th>
              </tr>
            </thead>
            <tbody>
              {chunks.map((chunk) => (
                <tr key={chunk.id} data-testid="chunk">
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`Select ${chunk.id}`}
                      checked={selected.has(chunk.id)}
                      onChange={() => toggle(chunk.id)}
                    />
                  </td>
                  <td>{String(chunk.metadata.source_path ?? '')}</td>
                  <td>{String(chunk.metadata.course ?? '')}</td>
                  <td>{String(chunk.metadata.unit ?? '')}</td>
                  <td>
                    <button
                      type="button"
                      className="link"
                      onClick={() => setInspected(inspected === chunk.id ? null : chunk.id)}
                    >
                      {inspected === chunk.id ? 'Hide' : 'Inspect'}
                    </button>{' '}
                    {inspected === chunk.id ? <pre>{chunk.text}</pre> : chunk.text.slice(0, 80)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="row">
            <button
              type="button"
              disabled={offset === 0}
              onClick={() => setOffset(offset - PAGE_SIZE)}
            >
              Previous
            </button>
            <span className="muted">
              {offset + 1}–{offset + chunks.length}
            </span>
            <button type="button" disabled={!hasNext} onClick={() => setOffset(offset + PAGE_SIZE)}>
              Next
            </button>
          </div>
        </>
      )}

      {plan && (
        <div className="panel" role="dialog" aria-label="Confirm delete">
          {plan.report.matched === 0 ? (
            <p>Nothing matches, so there is nothing to delete.</p>
          ) : (
            <p>
              This will permanently delete <strong>{plan.report.matched}</strong> chunk
              {plan.report.matched === 1 ? '' : 's'} from both indexes.
            </p>
          )}
          <div className="row">
            <button
              type="button"
              disabled={plan.report.matched === 0}
              onClick={() => void confirm()}
            >
              Delete
            </button>
            <button type="button" onClick={() => setPlan(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
