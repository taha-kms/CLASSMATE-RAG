import { useRef, useState } from 'react';
import { ApiError, askStream, getChunk } from './api';
import { FilterFields } from './FilterFields';
import { RichText } from './rich';
import type { Knobs } from './settings-store';
import type { AskResult, Chunk, Filters, Stage } from './types';

export default function Chat({ knobs }: { knobs: Knobs }) {
  const [question, setQuestion] = useState('');
  const [filters, setFilters] = useState<Filters>({});
  const [text, setText] = useState('');
  const [stage, setStage] = useState<Stage | null>(null);
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [opened, setOpened] = useState<Chunk | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || pending) return;

    controller.current = new AbortController();
    setPending(true);
    setError(null);
    setResult(null);
    setText('');
    setStage(null);
    setOpened(null);

    try {
      const final = await askStream(
        question,
        { ...filters, ...knobs },
        {
          onStage: setStage,
          onToken: (token) => setText((current) => current + token),
          onReplace: () => setText(''),
        },
        controller.current.signal,
      );
      setResult(final);
      setText(final.answer);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') {
        setError('Stopped.');
      } else {
        setError(caught instanceof ApiError ? caught.message : 'Something went wrong.');
      }
    } finally {
      setPending(false);
      setStage(null);
    }
  }

  async function openCitation(n: number) {
    setOpenError(null);
    const id = result?.retrieved[n - 1]?.id;
    if (!id) {
      setOpenError(`No chunk is recorded for [${n}].`);
      return;
    }
    try {
      setOpened(await getChunk(id));
    } catch (caught) {
      setOpenError(caught instanceof ApiError ? caught.message : 'Could not open that chunk.');
    }
  }

  return (
    <section aria-label="Chat">
      <form onSubmit={onSubmit} className="ask">
        <label htmlFor="question" className="ask-label">
          Question
        </label>
        <input
          id="question"
          name="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="What is the chain rule?"
          autoComplete="off"
        />
        <details>
          <summary>Filters</summary>
          <FilterFields value={filters} onChange={setFilters} />
        </details>
        <div className="row">
          <button type="submit" disabled={pending || !question.trim()}>
            {pending ? 'Asking…' : 'Ask'}
          </button>
          {pending && (
            <button type="button" onClick={() => controller.current?.abort()}>
              Stop
            </button>
          )}
        </div>
      </form>

      {!pending && !text && !result && !error && (
        <p className="empty">
          Ask about anything you have ingested. When an answer comes from your notes, each claim
          links to the passage behind it.
        </p>
      )}

      {pending && stage && (
        <p className="muted" role="status" data-testid="stage">
          {stage.message}
        </p>
      )}

      {error && <p role="alert">{error}</p>}

      {(text || result) && (
        <article aria-label="answer" className="page">
          <p className="answer">
            <RichText text={text} onCite={result ? openCitation : undefined} />
          </p>

          {result && result.sources.length > 0 && (
            <div className="margin">
              <h2>Sources</h2>
              <ol>
                {result.sources.map((source) => (
                  <li key={source.n} data-testid="source">
                    <button type="button" className="cite" onClick={() => openCitation(source.n)}>
                      [{source.n}]
                    </button>{' '}
                    {source.ref}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {result?.notice && <p data-testid="notice">{result.notice}</p>}
        </article>
      )}

      {openError && <p role="alert">{openError}</p>}

      {opened && (
        <aside aria-label="cited chunk" className="panel">
          <div className="row">
            <strong>{String(opened.metadata.source_path ?? opened.id)}</strong>
            <button type="button" onClick={() => setOpened(null)}>
              Close
            </button>
          </div>
          <pre>
            <RichText text={opened.text} />
          </pre>
        </aside>
      )}
    </section>
  );
}
