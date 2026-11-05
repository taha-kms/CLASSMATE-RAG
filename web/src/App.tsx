import { useEffect, useState } from 'react';
import Chat from './Chat';
import Corpus from './Corpus';
import Ingest from './Ingest';
import Preview from './Preview';
import Settings from './Settings';
import { useKnobs } from './settings-store';

const VIEWS = ['chat', 'preview', 'ingest', 'corpus', 'settings'] as const;
type View = (typeof VIEWS)[number];

const LABELS: Record<View, string> = {
  chat: 'Chat',
  preview: 'Preview',
  ingest: 'Ingest',
  corpus: 'Corpus',
  settings: 'Settings',
};

/**
 * The view lives in the URL hash, not the path. Paths such as /chunks and
 * /stats are API routes on the same server, so reloading on one of them would
 * return JSON instead of this page.
 */
function viewFromHash(): View {
  const name = window.location.hash.replace(/^#\/?/, '');
  return (VIEWS as readonly string[]).includes(name) ? (name as View) : 'chat';
}

export default function App() {
  const [view, setView] = useState<View>(viewFromHash);
  const [knobs, setKnobs] = useKnobs();

  useEffect(() => {
    const onHashChange = () => setView(viewFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  return (
    <main>
      <h1>CLASSMATE-RAG</h1>
      <nav aria-label="Views">
        {VIEWS.map((name) => (
          <a key={name} href={`#/${name}`} aria-current={name === view ? 'page' : undefined}>
            {LABELS[name]}
          </a>
        ))}
      </nav>

      {view === 'chat' && <Chat knobs={knobs} />}
      {view === 'preview' && <Preview knobs={knobs} />}
      {view === 'ingest' && <Ingest />}
      {view === 'corpus' && <Corpus />}
      {view === 'settings' && <Settings knobs={knobs} onKnobs={setKnobs} />}
    </main>
  );
}
