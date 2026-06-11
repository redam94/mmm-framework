import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Check, Loader2, Lock, Plus, Save, X } from 'lucide-react';
import { Modal } from '../common/Modal';
import { Badge } from '../common/Badge';
import { FLabel, iCls, sCls } from '../common/form';
import { API_BASE, authHeaders } from '../../constants';
import { BrandPreview } from './BrandPreview';
import { ExtractFromWebsite } from './ExtractFromWebsite';
import { PalettePicker } from './PalettePicker';
import {
  FONT_SUGGESTIONS, MAX_PALETTE, brandingToForm, emptyBrandingForm,
  formToBranding, isHex, matchPresetId, presetToBranding, toPickerHex,
  PALETTE_PRESETS,
  type Branding, type BrandingForm, type PalettePreset,
} from './lib';

const FONT_LIST_ID = 'branding-font-suggestions';
const CURRENCIES = ['USD', 'EUR', 'GBP'] as const;

// ─── Small field primitives (internal) ───────────────────────────────────────

function HexField({ label, value, onChange, disabled }: {
  label: string; value: string; onChange: (v: string) => void; disabled?: boolean;
}) {
  const invalid = value.trim() !== '' && !isHex(value);
  return (
    <div>
      <FLabel>{label}</FLabel>
      <div className="flex items-center gap-1.5">
        <input
          type="color"
          value={toPickerHex(value)}
          disabled={disabled}
          onChange={e => onChange(e.target.value)}
          className="w-8 h-8 p-0.5 rounded-lg border border-gray-200 bg-white cursor-pointer shrink-0 disabled:opacity-40"
        />
        <input
          type="text"
          value={value}
          disabled={disabled}
          onChange={e => onChange(e.target.value)}
          placeholder="#rrggbb"
          className={`${iCls} font-mono ${invalid ? 'border-red-300 focus:ring-red-300' : ''}`}
        />
      </div>
      {invalid && <p className="text-[10px] text-red-500 mt-0.5">Not a hex color</p>}
    </div>
  );
}

function PaletteEditor({ palette, onChange, disabled }: {
  palette: string[]; onChange: (p: string[]) => void; disabled?: boolean;
}) {
  return (
    <div>
      <FLabel>Palette ({palette.length}/{MAX_PALETTE})</FLabel>
      <div className="flex items-center gap-1.5 flex-wrap">
        {palette.map((c, i) => (
          <span key={i} className="relative group">
            <input
              type="color"
              value={toPickerHex(c)}
              disabled={disabled}
              title={c}
              onChange={e => onChange(palette.map((x, j) => (j === i ? e.target.value : x)))}
              className="w-8 h-8 p-0.5 rounded-lg border border-gray-200 bg-white cursor-pointer disabled:opacity-40"
            />
            {!disabled && (
              <button
                type="button"
                onClick={() => onChange(palette.filter((_, j) => j !== i))}
                className="absolute -top-1.5 -right-1.5 hidden group-hover:flex items-center justify-center w-4 h-4 rounded-full bg-gray-700 text-white"
                title="Remove color"
              >
                <X size={9} />
              </button>
            )}
          </span>
        ))}
        <button
          type="button"
          disabled={disabled || palette.length >= MAX_PALETTE}
          onClick={() => onChange([...palette, palette[palette.length - 1] || '#8fa86a'])}
          className="flex items-center justify-center w-8 h-8 rounded-lg border border-dashed border-gray-300 text-gray-400 hover:border-indigo-400 hover:text-indigo-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          title="Add color"
        >
          <Plus size={14} />
        </button>
      </div>
    </div>
  );
}

// ─── Main modal ───────────────────────────────────────────────────────────────

export function BrandingModal({ projectId, apiKey, modelName, onClose }: {
  projectId: string | null;
  apiKey: string | null;
  modelName: string | null;
  onClose: () => void;
}) {
  const [section, setSection] = useState<'project' | 'global'>('project');

  // ── Project branding state ──
  const [form, setForm] = useState<BrandingForm>(emptyBrandingForm());
  const [loadedConfirmed, setLoadedConfirmed] = useState<boolean | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ kind: 'success' | 'error'; text: string } | null>(null);
  const [savedNote, setSavedNote] = useState(false);

  // ── Global defaults state ──
  const [globalPresetId, setGlobalPresetId] = useState<'' | PalettePreset['id']>('');
  const [currency, setCurrency] = useState<string>('USD');
  const [globalDirty, setGlobalDirty] = useState(false);
  const [globalSaving, setGlobalSaving] = useState(false);
  const [globalMsg, setGlobalMsg] = useState<{ kind: 'success' | 'error'; text: string } | null>(null);
  const [hostedDisabled, setHostedDisabled] = useState(false);

  // Load current project branding on open / project switch.
  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    (async () => {
      try {
        const data: Branding = await fetch(
          `${API_BASE}/projects/${encodeURIComponent(projectId)}/branding`,
          { headers: authHeaders(apiKey, modelName) },
        ).then(r => r.json());
        if (cancelled || !data || typeof data !== 'object') return;
        setForm(brandingToForm(data));
        setLoadedConfirmed(Object.keys(data).length > 0 ? data.confirmed !== false : null);
        setDirty(false);
      } catch { /* leave the empty form */ }
    })();
    return () => { cancelled = true; };
  }, [projectId, apiKey, modelName]);

  // Load global preferences on open.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetch(`${API_BASE}/preferences`, { headers: authHeaders(apiKey, modelName) })
          .then(r => r.json());
        if (cancelled) return;
        const prefs = (data?.preferences ?? {}) as Record<string, unknown>;
        if (typeof prefs.currency === 'string' && (CURRENCIES as readonly string[]).includes(prefs.currency)) {
          setCurrency(prefs.currency);
        }
        const defaults = prefs.branding_defaults as Branding | undefined;
        const matched = matchPresetId(defaults?.colors?.primary);
        if (matched) setGlobalPresetId(matched);
      } catch { /* defaults stay */ }
    })();
    return () => { cancelled = true; };
  }, [apiKey, modelName]);

  // Dirty-state guard on close (Esc / backdrop / X all funnel through here).
  const handleClose = useCallback(() => {
    if ((dirty || globalDirty) && !window.confirm('Discard unsaved branding changes?')) return;
    onClose();
  }, [dirty, globalDirty, onClose]);

  const update = useCallback((patch: Partial<BrandingForm>) => {
    setForm(prev => ({ ...prev, ...patch }));
    setDirty(true);
    setSaveMsg(null);
    setSavedNote(false);
  }, []);

  const applyPreset = useCallback((p: PalettePreset) => {
    update({ primary: p.primary, secondary: p.secondary, accent: p.accent, palette: [...p.palette] });
  }, [update]);

  const confirmProposal = useCallback((proposal: Branding) => {
    setForm(brandingToForm(proposal));
    setLoadedConfirmed(false);
    setDirty(true);
    setSaveMsg(null);
    setSavedNote(false);
  }, []);

  const saveProject = async () => {
    if (!projectId || saving) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/branding`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify(formToBranding(form)),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setSaveMsg({
          kind: 'error',
          text: typeof data?.detail === 'string' ? data.detail : `Save failed (HTTP ${res.status})`,
        });
        return;
      }
      setForm(brandingToForm(data as Branding));
      setLoadedConfirmed(true);
      setDirty(false);
      setSaveMsg({ kind: 'success', text: 'Branding saved.' });
      setSavedNote(true);
    } catch {
      setSaveMsg({ kind: 'error', text: 'Save failed — is the API running?' });
    } finally {
      setSaving(false);
    }
  };

  const saveGlobal = async () => {
    if (globalSaving || hostedDisabled) return;
    setGlobalSaving(true);
    setGlobalMsg(null);
    try {
      const preferences: Record<string, unknown> = { currency };
      const preset = PALETTE_PRESETS.find(p => p.id === globalPresetId);
      if (preset) preferences.branding_defaults = presetToBranding(preset);
      const res = await fetch(`${API_BASE}/preferences`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ preferences }),
      });
      if (res.status === 403) {
        setHostedDisabled(true);
        setGlobalDirty(false);
        return;
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setGlobalMsg({
          kind: 'error',
          text: typeof data?.detail === 'string' ? data.detail : `Save failed (HTTP ${res.status})`,
        });
        return;
      }
      setGlobalDirty(false);
      setGlobalMsg({ kind: 'success', text: 'Global defaults saved.' });
    } catch {
      setGlobalMsg({ kind: 'error', text: 'Save failed — is the API running?' });
    } finally {
      setGlobalSaving(false);
    }
  };

  const previewBranding = formToBranding(form);

  return (
    <Modal title="Branding & Preferences" onClose={handleClose}>
      <datalist id={FONT_LIST_ID}>
        {FONT_SUGGESTIONS.map(f => <option key={f} value={f} />)}
      </datalist>

      {/* Segmented control */}
      <div className="inline-flex rounded-lg border border-gray-200 bg-gray-100 p-0.5 mb-5">
        {([['project', 'Project branding'], ['global', 'Global defaults']] as const).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setSection(id)}
            className={`px-3.5 py-1.5 text-sm font-medium rounded-md transition-colors ${
              section === id ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Project branding section ── */}
      {section === 'project' && (
        !projectId ? (
          <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-6 text-center">
            <p className="text-sm text-gray-500">
              No active project. Select or create a project in the sidebar to set its branding.
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {loadedConfirmed === false && (
              <div className="flex items-center gap-2">
                <Badge label="Proposed (unconfirmed)" color="amber" />
                <span className="text-xs text-gray-400">
                  Unconfirmed branding never styles output — review and Save to confirm.
                </span>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Left: form */}
              <div className="space-y-3">
                <div>
                  <FLabel>Client name</FLabel>
                  <input
                    type="text" value={form.client_name}
                    onChange={e => update({ client_name: e.target.value })}
                    placeholder="Acme Corp" className={iCls}
                  />
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <HexField label="Primary" value={form.primary} onChange={v => update({ primary: v })} />
                  <HexField label="Secondary" value={form.secondary} onChange={v => update({ secondary: v })} />
                  <HexField label="Accent" value={form.accent} onChange={v => update({ accent: v })} />
                </div>

                <PaletteEditor palette={form.palette} onChange={p => update({ palette: p })} />

                <div>
                  <FLabel>Presets</FLabel>
                  <PalettePicker activePrimary={form.primary} onApply={applyPreset} />
                </div>

                <div>
                  <FLabel>Logo URL</FLabel>
                  <input
                    type="url" value={form.logo_url}
                    onChange={e => update({ logo_url: e.target.value })}
                    placeholder="https://client.com/logo.png" className={iCls}
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <FLabel>Heading font</FLabel>
                    <input
                      type="text" value={form.font_heading} list={FONT_LIST_ID}
                      onChange={e => update({ font_heading: e.target.value })}
                      placeholder="Inter" className={iCls}
                    />
                  </div>
                  <div>
                    <FLabel>Body font</FLabel>
                    <input
                      type="text" value={form.font_body} list={FONT_LIST_ID}
                      onChange={e => update({ font_body: e.target.value })}
                      placeholder="Inter" className={iCls}
                    />
                  </div>
                </div>

                <div>
                  <FLabel>Footer text</FLabel>
                  <input
                    type="text" value={form.footer_text}
                    onChange={e => update({ footer_text: e.target.value })}
                    placeholder="Confidential — prepared for Acme Corp" className={iCls}
                  />
                </div>
              </div>

              {/* Right: extract + live preview */}
              <div className="space-y-4">
                <div>
                  <FLabel>Extract from website</FLabel>
                  <ExtractFromWebsite
                    projectId={projectId}
                    apiKey={apiKey}
                    modelName={modelName}
                    onConfirm={confirmProposal}
                  />
                </div>
                <div>
                  <FLabel>Preview</FLabel>
                  <BrandPreview branding={previewBranding} />
                </div>
              </div>
            </div>

            {/* Save row */}
            <div className="flex items-center gap-3 pt-2 border-t border-gray-100">
              <button
                onClick={saveProject}
                disabled={saving || !dirty}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-500 disabled:opacity-40 transition-colors"
              >
                {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />} Save branding
              </button>
              {saveMsg && (
                <span className={`flex items-center gap-1.5 text-xs ${saveMsg.kind === 'success' ? 'text-emerald-600' : 'text-red-600'}`}>
                  {saveMsg.kind === 'success' ? <Check size={13} /> : <AlertTriangle size={13} />} {saveMsg.text}
                </span>
              )}
              {dirty && !saveMsg && <span className="text-xs text-gray-400">Unsaved changes</span>}
            </div>
            {savedNote && (
              <p className="text-xs text-indigo-700 bg-indigo-50 border border-indigo-200 rounded-lg px-3 py-2">
                New plots and reports will use this branding; existing plots keep their colors.
              </p>
            )}
          </div>
        )
      )}

      {/* ── Global defaults section ── */}
      {section === 'global' && (
        <div className="space-y-4 max-w-md">
          {hostedDisabled && (
            <p className="flex items-center gap-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
              <Lock size={13} className="shrink-0" /> Global preferences are disabled in hosted mode. Use per-project branding instead.
            </p>
          )}

          <div>
            <FLabel>Favorite palette</FLabel>
            <PalettePicker
              activePrimary={PALETTE_PRESETS.find(p => p.id === globalPresetId)?.primary ?? ''}
              onApply={p => { setGlobalPresetId(p.id); setGlobalDirty(true); setGlobalMsg(null); }}
              disabled={hostedDisabled}
            />
            <p className="text-[10px] text-gray-400 mt-1">
              Used as the default branding for projects without their own.
            </p>
          </div>

          <div>
            <FLabel>Currency</FLabel>
            <select
              value={currency}
              disabled={hostedDisabled}
              onChange={e => { setCurrency(e.target.value); setGlobalDirty(true); setGlobalMsg(null); }}
              className={`${sCls} disabled:opacity-40`}
            >
              {CURRENCIES.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className="flex items-center gap-3 pt-2 border-t border-gray-100">
            <button
              onClick={saveGlobal}
              disabled={globalSaving || hostedDisabled || !globalDirty}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-500 disabled:opacity-40 transition-colors"
            >
              {globalSaving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />} Save defaults
            </button>
            {globalMsg && (
              <span className={`flex items-center gap-1.5 text-xs ${globalMsg.kind === 'success' ? 'text-emerald-600' : 'text-red-600'}`}>
                {globalMsg.kind === 'success' ? <Check size={13} /> : <AlertTriangle size={13} />} {globalMsg.text}
              </span>
            )}
          </div>
        </div>
      )}
    </Modal>
  );
}
