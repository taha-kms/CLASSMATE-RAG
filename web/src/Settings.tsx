import { useEffect, useRef, useState } from 'react';
import { ApiError, getJob, getStats, reconcile, startJob, vacuum } from './api';
import ServerSettings from './ServerSettings';
import type { Knobs } from './settings-store';
import type { IndexStats, Job, ReconcileReport } from './types';

const megabytes = (bytes: number) => `${(bytes / 1_048_576).toFixed(1)} MB`;
const messageOf = (caught: unknown) =>
  caught instanceof ApiError ? caught.message : 'Something went wrong.';

export default function Settings({
  knobs,
  onKnobs,
}: {
  knobs: Knobs;
  onKnobs: (next: Knobs) => void;
}) {
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [drift, setDrift] = useState<ReconcileReport | null>(null);

  async function refresh() {
    try {
      setStats(await getStats());
    } catch (caught) {
      setError(messageOf(caught));
    }
  }

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch((caught: unknown) => setError(messageOf(caught)));
  }, []);

  async function check() {
    setError(null);
    setNotice(null);
    try {
      setDrift(await reconcile(false));
    } catch (caught) {
      setError(messageOf(caught));
    }
  }

  async function repair() {
    setError(null);
    try {
      const done = await reconcile(true);
      setNotice(`Removed ${done.removed} stranded vector row${done.removed === 1 ? '' : 's'}.`);
      setDrift(null);
      await refresh();
    } catch (caught) {
      setError(messageOf(caught));
    }
  }

  async function compact() {
    setError(null);
    try {
      const outcome = await vacuum();
      setNotice(
        Object.entries(outcome)
          .map(([store, state]) => `${store}: ${state}`)
          .join(', '),
      );
      await refresh();
    } catch (caught) {
      setError(messageOf(caught));
    }
  }

  return (
    <section aria-label="Settings">
      <h2>Per-question retrieval</h2>
      <div className="filters">
        <label>
          Chunks per answer (top_k)
          <input
            type="number"
            min={1}
            max={100}
            value={knobs.top_k}
            onChange={(event) => {
              const next = Math.round(Number(event.target.value));
              if (next >= 1 && next <= 100) onKnobs({ ...knobs, top_k: next });
            }}
          />
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={knobs.hybrid}
            onChange={(event) => onKnobs({ ...knobs, hybrid: event.target.checked })}
          />
          Hybrid retrieval (keyword + vector)
        </label>
      </div>
      <p className="muted">Kept in this browser and sent with each question.</p>

      <h2>Server settings</h2>
      <ServerSettings />

      <h2>Index</h2>
      {error && <p role="alert">{error}</p>}
      {notice && <p role="status">{notice}</p>}
      {stats && (
        <dl data-testid="stats">
          <dt>Chunks</dt>
          <dd>{stats.vector_count}</dd>
          <dt>Stores agree</dt>
          <dd>{stats.consistent ? 'Yes' : 'No — run a consistency check'}</dd>
          <dt>Embedding model</dt>
          <dd>{stats.embedding_model}</dd>
          <dt>Vector store</dt>
          <dd>
            {stats.chroma.persist_dir} ({megabytes(stats.chroma.disk_bytes)})
          </dd>
          <dt>Keyword store</dt>
          <dd>
            {stats.bm25.persist_dir} ({megabytes(stats.bm25.disk_bytes)})
          </dd>
        </dl>
      )}

      <h2>Maintenance</h2>
      <div className="row">
        <button type="button" onClick={() => void check()}>
          Check consistency
        </button>
        <button type="button" onClick={() => void compact()}>
          Vacuum
        </button>
      </div>
      {drift && (
        <div className="panel" role="status">
          {drift.vector_only + drift.bm25_only === 0 ? (
            <p>Both stores agree.</p>
          ) : (
            <>
              <p>
                {drift.vector_only} row{drift.vector_only === 1 ? '' : 's'} only in the vector
                store, {drift.bm25_only} only in the keyword store.
              </p>
              <button type="button" onClick={() => void repair()}>
                Remove the vector-only rows
              </button>
            </>
          )}
        </div>
      )}

      <JobForm operation="dump" label="Back up" field="path" fieldLabel="Write to (file path)" />
      <JobForm
        operation="restore"
        label="Restore"
        field="path"
        fieldLabel="Read from (file path)"
      />
      <JobForm
        operation="rebuild"
        label="Rebuild embeddings"
        field="model"
        fieldLabel="Embedding model"
        warning="Re-embeds every chunk. This takes minutes."
      />
    </section>
  );
}

function JobForm({
  operation,
  label,
  field,
  fieldLabel,
  warning,
}: {
  operation: 'dump' | 'restore' | 'rebuild';
  label: string;
  field: 'path' | 'model';
  fieldLabel: string;
  warning?: string;
}) {
  const [value, setValue] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [lost, setLost] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  function poll(id: string) {
    timer.current = window.setTimeout(async () => {
      try {
        const latest = await getJob(id);
        if (latest === null) {
          // Jobs live in memory: a 404 means the server restarted under us.
          setLost(true);
          return;
        }
        setJob(latest);
        if (latest.state === 'running') poll(id);
      } catch (caught) {
        setError(messageOf(caught));
      }
    }, 1500);
  }

  async function start(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setLost(false);
    setJob(null);
    try {
      const started = await startJob(operation, { [field]: value });
      setJob(started);
      poll(started.id);
    } catch (caught) {
      setError(messageOf(caught));
    }
  }

  const running = job?.state === 'running';

  return (
    <form onSubmit={start} className="job" aria-label={label}>
      <h3>{label}</h3>
      {warning && <p className="muted">{warning}</p>}
      <label>
        {fieldLabel}
        <input
          value={value}
          onChange={(event) => setValue(event.target.value)}
          autoComplete="off"
        />
      </label>
      <button type="submit" disabled={!value.trim() || running}>
        {running ? 'Running…' : label}
      </button>
      {error && <p role="alert">{error}</p>}
      {lost && (
        <p role="alert">
          The server no longer knows this job. It was probably restarted, so start it again.
        </p>
      )}
      {job && !running && (
        <p role="status">
          {job.state === 'succeeded'
            ? `Done: ${JSON.stringify(job.result)}`
            : `Failed: ${job.error ?? 'see the server log'}`}
        </p>
      )}
    </form>
  );
}
