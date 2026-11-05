import { useEffect, useState } from 'react';
import { ApiError, getSettings, putSettings } from './api';
import type { SettingChange, SettingInfo } from './types';

const messageOf = (caught: unknown) =>
  caught instanceof ApiError ? caught.message : 'Something went wrong.';

/** Where a value came from, in words a student would use. */
const SOURCE_LABEL: Record<SettingInfo['source'], string> = {
  environment: 'set in the server’s environment',
  file: 'set in .env',
  default: 'default',
  stored: 'stored',
  unset: 'not set',
};

function Field({
  setting,
  draft,
  onChange,
}: {
  setting: SettingInfo;
  draft: string | null | undefined;
  onChange: (next: string | null | undefined) => void;
}) {
  const id = `setting-${setting.key}`;

  if (setting.local_only) {
    return (
      <div>
        <input id={id} type="text" value={setting.value} readOnly aria-readonly="true" />
        <p className="muted">
          Read-only here, because it decides where your data is sent. Change it on this machine with{' '}
          <code>rag config set {setting.key}=VALUE</code>.
        </p>
      </div>
    );
  }
  const current = draft === undefined ? setting.value : (draft ?? '');
  // Empty text means "use the default", so show the default as the hint.
  const placeholder = setting.default || undefined;

  if (setting.kind === 'secret') {
    return (
      <div>
        <input
          id={id}
          type="password"
          autoComplete="off"
          placeholder={setting.source === 'unset' ? 'Enter a key' : 'Enter a new key to replace it'}
          value={typeof draft === 'string' ? draft : ''}
          onChange={(event) => onChange(event.target.value === '' ? undefined : event.target.value)}
        />
        <p className="muted">
          {setting.source === 'unset'
            ? 'Not set.'
            : `Configured (${setting.value}), ${SOURCE_LABEL[setting.source]}.`}{' '}
          {setting.source === 'stored' && (
            <button
              type="button"
              className="link"
              onClick={() => onChange(draft === null ? undefined : null)}
            >
              {draft === null ? 'Keep it' : 'Remove it'}
            </button>
          )}
          {draft === null && ' Will be removed when you save.'}
        </p>
      </div>
    );
  }

  if (setting.kind === 'bool') {
    return (
      <label className="check">
        <input
          id={id}
          type="checkbox"
          checked={current === 'true'}
          onChange={(event) => onChange(event.target.checked ? 'true' : 'false')}
        />
        {current === 'true' ? 'On' : 'Off'}
      </label>
    );
  }

  if (setting.kind === 'choice') {
    return (
      <select id={id} value={current} onChange={(event) => onChange(event.target.value)}>
        {setting.choices.map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    );
  }

  const numeric = setting.kind === 'int' || setting.kind === 'float';
  return (
    <input
      id={id}
      type={numeric ? 'number' : 'text'}
      step={setting.kind === 'float' ? 'any' : undefined}
      min={setting.minimum ?? undefined}
      max={setting.maximum ?? undefined}
      placeholder={placeholder}
      autoComplete="off"
      value={current}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

/**
 * Every server setting, grouped, with where each value came from.
 *
 * Only fields the user touched are sent. A credential field starts empty and
 * stays empty after saving: the server never returns the key, so there is
 * nothing to pre-fill and nothing to leak into a screenshot.
 */
export default function ServerSettings() {
  const [settings, setSettings] = useState<SettingInfo[] | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string | null | undefined>>({});
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<SettingChange[] | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getSettings()
      .then(setSettings)
      .catch((caught: unknown) => setError(messageOf(caught)));
  }, []);

  const changes = Object.fromEntries(
    Object.entries(drafts).filter(([key, value]) => {
      if (value === undefined) return false;
      const setting = settings?.find((s) => s.key === key);
      return setting !== undefined && (value === null || value !== setting.value);
    }),
  ) as Record<string, string | null>;
  const dirty = Object.keys(changes).length;

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (dirty === 0) return;
    setSaving(true);
    setError(null);
    setSaved(null);
    try {
      const results = await putSettings(changes);
      // Refetch before clearing the drafts: clearing first shows the old
      // value for a moment, as if the save had not taken.
      setSettings(await getSettings());
      setDrafts({});
      setSaved(results);
    } catch (caught) {
      // Rejected as a whole: nothing was written, so the drafts are kept to fix.
      setError(messageOf(caught));
    } finally {
      setSaving(false);
    }
  }

  const groups = [...new Set(settings?.map((s) => s.group) ?? [])];

  return (
    <form onSubmit={save} aria-label="Server settings">
      {error && <p role="alert">{error}</p>}
      {saved && (
        <div className="panel" role="status">
          <strong>Saved.</strong>
          <ul>
            {saved.map((change) => (
              <li key={change.key}>
                {change.key}: {change.action === 'unset' ? 'removed' : 'updated'}
                {change.note ? ` — ${change.note}` : change.applied ? ' — in effect now' : ''}
              </li>
            ))}
          </ul>
        </div>
      )}

      {groups.map((group) => (
        <details key={group} open={group === groups[0]}>
          <summary>{group}</summary>
          {settings
            ?.filter((s) => s.group === group)
            .map((setting) => (
              <div key={setting.key} className="setting" data-testid={`setting-${setting.key}`}>
                <label htmlFor={`setting-${setting.key}`}>{setting.key}</label>
                <Field
                  setting={setting}
                  draft={drafts[setting.key]}
                  onChange={(next) => setDrafts({ ...drafts, [setting.key]: next })}
                />
                <p className="muted">
                  {setting.description}
                  {setting.kind !== 'secret' && ` (${SOURCE_LABEL[setting.source]})`}
                  {setting.restart_required && ' Needs a restart to take effect.'}
                </p>
                {setting.overridden && (
                  <p role="note">
                    The server is using <code>{setting.value}</code>, from its environment, while{' '}
                    <code>.env</code> says <code>{setting.file_value}</code>. That is a variable
                    exported in your shell, or a change made outside this page since the server
                    started. A value saved here will not be used until you unset that variable or
                    restart the server.
                  </p>
                )}
              </div>
            ))}
        </details>
      ))}

      <div className="row">
        <button type="submit" disabled={dirty === 0 || saving}>
          {saving
            ? 'Saving…'
            : dirty === 0
              ? 'Save'
              : `Save ${dirty} change${dirty === 1 ? '' : 's'}`}
        </button>
        {dirty > 0 && (
          <button type="button" onClick={() => setDrafts({})}>
            Discard
          </button>
        )}
      </div>
    </form>
  );
}
