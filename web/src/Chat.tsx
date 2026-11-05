import { Fragment, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { ApiError, askStream, getChunk } from './api';
import { FilterFields } from './FilterFields';
import type { Knobs } from './settings-store';
import type { AskResult, Chunk, Filters, Stage } from './types';

/** Turn [n] markers in an answer into buttons that open the cited chunk. */
function withCitations(text: string, onOpen: (n: number) => void): ReactNode[] {
  return text.split(/(\[\d+\])/g).map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match) return <Fragment key={index}>{part}</Fragment>;
    const n = Number(match[1]);
    return (
      <button key={index} type="button" className="cite" onClick={() => onOpen(n)}>
        {part}
      </button>
    );
  });
}

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

      {pending && stage && (
        <p className="muted" role="status" data-testid="stage">
          {stage.message}
        </p>
      )}

      {error && <p role="alert">{error}</p>}

      {(text || result) && (
        <article aria-label="answer">
          <p className="answer">{result ? withCitations(text, openCitation) : text}</p>

          {result?.notice && (
            <p className="muted" data-testid="notice">
              {result.notice}
            </p>
          )}

          {result && result.sources.length > 0 && (
            <>
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
            </>
          )}
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
          <pre>{opened.text}</pre>
        </aside>
      )}
    </section>
  );
}
