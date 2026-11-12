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

/** What each page is for, in a line. Shown under the title so no page opens blank. */
const HEAD: Record<View, { title: string; blurb: string }> = {
  chat: {
    title: 'What would you like to know?',
    blurb: 'Answers come from your own notes, with the passage behind each claim.',
  },
  preview: {
    title: 'See what gets retrieved',
    blurb: 'Retrieval only, no model. Use it to find out why an answer went wrong.',
  },
  ingest: {
    title: 'Add your notes',
    blurb: 'Upload a file and tag it, so you can filter by course and unit later.',
  },
  corpus: {
    title: 'Everything in your index',
    blurb: 'Browse the pieces your notes were split into, and remove what you no longer want.',
  },
  settings: {
    title: 'Settings',
    blurb: 'How answers are written, and where your questions and notes are sent.',
  },
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
    <div className="shell">
      <header className="topbar">
        <h1>Classmate</h1>
        <nav aria-label="Views">
          {VIEWS.map((name) => (
            <a key={name} href={`#/${name}`} aria-current={name === view ? 'page' : undefined}>
              {LABELS[name]}
            </a>
          ))}
        </nav>
      </header>
      <main data-view={view}>
        <div className="page-head">
          <h2>{HEAD[view].title}</h2>
          <p>{HEAD[view].blurb}</p>
        </div>
        {view === 'chat' && <Chat knobs={knobs} />}
        {view === 'preview' && <Preview knobs={knobs} />}
        {view === 'ingest' && <Ingest />}
        {view === 'corpus' && <Corpus />}
        {view === 'settings' && <Settings knobs={knobs} onKnobs={setKnobs} />}
      </main>
    </div>
  );
}
