import { useRef, useState } from 'react';
import { ApiError, ingest } from './api';
import type { IngestMeta } from './types';

const FIELDS: Array<[keyof IngestMeta, string]> = [
  ['course', 'Course'],
  ['unit', 'Unit'],
  ['language', 'Language'],
  ['doc_type', 'Type'],
  ['author', 'Author'],
  ['semester', 'Semester'],
  ['tags', 'Tags (comma separated)'],
];

export default function Ingest() {
  const [file, setFile] = useState<File | null>(null);
  const [meta, setMeta] = useState<IngestMeta>({});
  const [dragging, setDragging] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const picker = useRef<HTMLInputElement>(null);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    setPending(true);
    setError(null);
    setResult(null);
    try {
      setResult(await ingest(file, meta));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Something went wrong.');
    } finally {
      setPending(false);
    }
  }

  return (
    <section aria-label="Ingest">
      <form onSubmit={onSubmit}>
        <div
          className={`drop${dragging ? ' dragging' : ''}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            setFile(event.dataTransfer.files[0] ?? null);
          }}
        >
          <p>{file ? file.name : 'Drop a file here, or'}</p>
          <label htmlFor="file">File</label>
          <input
            id="file"
            type="file"
            ref={picker}
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </div>

        <div className="filters">
          {FIELDS.map(([key, label]) => (
            <label key={key}>
              {label}
              <input
                value={meta[key] ?? ''}
                onChange={(event) => setMeta({ ...meta, [key]: event.target.value })}
                autoComplete="off"
              />
            </label>
          ))}
        </div>

        {/* The API answers once the file is indexed; there is no incremental
            progress to report, so say what is happening rather than fake a bar. */}
        <button type="submit" disabled={!file || pending}>
          {pending ? 'Ingesting… large files take a while' : 'Ingest'}
        </button>
      </form>

      {error && <p role="alert">{error}</p>}
      {result && (
        <div className="panel" role="status">
          <strong>Ingested.</strong>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </section>
  );
}
