import { MetaFields } from './MetaFields';
import type { Filters } from './types';

/** The metadata filters the CLI has. Empty means "do not filter on this". */
export function FilterFields({
  value,
  onChange,
}: {
  value: Filters;
  onChange: (next: Filters) => void;
}) {
  return <MetaFields value={value} onChange={onChange} />;
}

export const hasFilter = (filters: Filters): boolean =>
  Object.values(filters).some((v) => v !== undefined && v.trim() !== '');
