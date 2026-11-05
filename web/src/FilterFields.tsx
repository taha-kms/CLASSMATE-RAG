import type { Filters } from './types';

const FIELDS: Array<[keyof Filters, string]> = [
  ['course', 'Course'],
  ['unit', 'Unit'],
  ['language', 'Language'],
  ['doc_type', 'Type'],
  ['author', 'Author'],
  ['semester', 'Semester'],
];

/** The metadata filters the CLI has. Empty means "do not filter on this". */
export function FilterFields({
  value,
  onChange,
}: {
  value: Filters;
  onChange: (next: Filters) => void;
}) {
  return (
    <div className="filters">
      {FIELDS.map(([key, label]) => (
        <label key={key}>
          {label}
          <input
            value={value[key] ?? ''}
            onChange={(event) => onChange({ ...value, [key]: event.target.value })}
            autoComplete="off"
          />
        </label>
      ))}
    </div>
  );
}

export const hasFilter = (filters: Filters): boolean =>
  Object.values(filters).some((v) => v !== undefined && v.trim() !== '');
