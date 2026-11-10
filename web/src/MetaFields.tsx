import { useEffect, useId, useState } from 'react';
import { getFacets } from './api';
import type { Facets } from './types';

const ADD = '__add__';

type Key = 'course' | 'unit' | 'language' | 'author';

const LISTED: Array<[Extract<Key, keyof Facets>, string]> = [
  ['course', 'Course'],
  ['unit', 'Unit'],
];
const LISTED_AFTER: Array<[Extract<Key, keyof Facets>, string]> = [['author', 'Author']];

// The server only accepts these for language, so it is a fixed list.
const LANGUAGES = ['en', 'it'];

/** The values already in the corpus. Without them the lists are just empty. */
function useFacets(): Facets | null {
  const [facets, setFacets] = useState<Facets | null>(null);
  useEffect(() => {
    let live = true;
    getFacets()
      .then((found) => live && setFacets(found))
      .catch(() => live && setFacets(null));
    return () => {
      live = false;
    };
  }, []);
  return facets;
}

/**
 * A dropdown of the values already in use. When new values are allowed, the
 * last entry, "+ Add new…", turns the field into a text box for one.
 */
function Choice({
  label,
  value,
  options,
  empty,
  allowNew,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  empty: string;
  allowNew: boolean;
  onChange: (next: string) => void;
}) {
  const id = useId();
  const [adding, setAdding] = useState(false);
  // A value that is not in the list (just added, or from an older session)
  // still has to show, or the select would silently display something else.
  const shown = value && !options.includes(value) ? [...options, value] : options;

  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {adding ? (
        <div className="adding">
          <input
            id={id}
            value={value}
            placeholder={`New ${label.toLowerCase()}`}
            autoComplete="off"
            autoFocus
            onChange={(event) => onChange(event.target.value)}
          />
          <button
            type="button"
            className="link"
            onClick={() => {
              setAdding(false);
              onChange('');
            }}
          >
            Back to list
          </button>
        </div>
      ) : (
        <select
          id={id}
          value={value}
          onChange={(event) => {
            if (event.target.value === ADD) {
              setAdding(true);
              onChange('');
            } else {
              onChange(event.target.value);
            }
          }}
        >
          <option value="">{empty}</option>
          {shown.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
          {allowNew && <option value={ADD}>+ Add new…</option>}
        </select>
      )}
    </div>
  );
}

export type MetaValue = Partial<Record<Key, string>>;

/**
 * Course, unit, language and author as dropdowns.
 *
 * For filtering, a value is chosen from what is already in the corpus: asking
 * for a course that does not exist can only return nothing. For ingesting,
 * `allowNew` adds "+ Add new…" to the lists that are open-ended, because that
 * is where a new course comes into being. Language is a fixed list either
 * way: the server rejects anything else.
 */
export function MetaFields<T extends MetaValue>({
  value,
  onChange,
  allowNew = false,
}: {
  value: T;
  onChange: (next: T) => void;
  allowNew?: boolean;
}) {
  const facets = useFacets();
  const empty = allowNew ? 'Not set' : 'Any';
  const set = (key: Key) => (next: string) => onChange({ ...value, [key]: next });
  const listed = (key: Extract<Key, keyof Facets>, label: string) => (
    <Choice
      key={key}
      label={label}
      value={value[key] ?? ''}
      options={facets?.[key] ?? []}
      empty={empty}
      allowNew={allowNew}
      onChange={set(key)}
    />
  );

  return (
    <div className="filters">
      {LISTED.map(([key, label]) => listed(key, label))}
      <Choice
        label="Language"
        value={value.language ?? ''}
        options={allowNew ? [...LANGUAGES, 'auto'] : LANGUAGES}
        empty={empty}
        allowNew={false}
        onChange={set('language')}
      />
      {LISTED_AFTER.map(([key, label]) => listed(key, label))}
    </div>
  );
}
