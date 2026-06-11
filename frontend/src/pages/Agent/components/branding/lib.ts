// Shared types + helpers for the branding/preferences UI. Kept component-free
// so component files only export components (react-refresh friendly).

// ─── Branding shape (mirrors mmm_framework.agents.branding.Branding) ────────

export interface BrandColors {
  primary?: string | null;
  secondary?: string | null;
  accent?: string | null;
  palette?: string[];
}

export interface BrandFonts {
  heading?: string | null;
  body?: string | null;
}

export interface Branding {
  client_name?: string | null;
  colors?: BrandColors;
  logo_url?: string | null;
  fonts?: BrandFonts;
  footer_text?: string | null;
  source_url?: string | null;
  source?: string;
  confirmed?: boolean;
}

// Flat editable form state ('' = unset; converted to null on save).
export interface BrandingForm {
  client_name: string;
  primary: string;
  secondary: string;
  accent: string;
  palette: string[];
  logo_url: string;
  font_heading: string;
  font_body: string;
  footer_text: string;
  source_url: string;
}

export function emptyBrandingForm(): BrandingForm {
  return {
    client_name: '', primary: '', secondary: '', accent: '', palette: [],
    logo_url: '', font_heading: '', font_body: '', footer_text: '', source_url: '',
  };
}

export function brandingToForm(b: Branding | null | undefined): BrandingForm {
  const c = b?.colors ?? {};
  const f = b?.fonts ?? {};
  return {
    client_name: b?.client_name ?? '',
    primary: c.primary ?? '',
    secondary: c.secondary ?? '',
    accent: c.accent ?? '',
    palette: (c.palette ?? []).filter(Boolean),
    logo_url: b?.logo_url ?? '',
    font_heading: f.heading ?? '',
    font_body: f.body ?? '',
    footer_text: b?.footer_text ?? '',
    source_url: b?.source_url ?? '',
  };
}

export function formToBranding(form: BrandingForm): Branding {
  const opt = (v: string): string | null => (v.trim() ? v.trim() : null);
  return {
    client_name: opt(form.client_name),
    colors: {
      primary: opt(form.primary),
      secondary: opt(form.secondary),
      accent: opt(form.accent),
      palette: form.palette.map(c => c.trim()).filter(Boolean),
    },
    logo_url: opt(form.logo_url),
    fonts: { heading: opt(form.font_heading), body: opt(form.font_body) },
    footer_text: opt(form.footer_text),
    source_url: opt(form.source_url),
    source: 'manual',
    confirmed: true,
  };
}

// ─── Hex helpers ─────────────────────────────────────────────────────────────

const HEX_RE = /^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$/;

export function isHex(v: string): boolean {
  return HEX_RE.test(v.trim());
}

/** Value safe for <input type="color"> (needs #rrggbb): expands #abc, falls
 * back to a neutral gray when empty/invalid. */
export function toPickerHex(v: string, fallback = '#888888'): string {
  const s = v.trim();
  if (!HEX_RE.test(s)) return fallback;
  if (s.length === 7) return s.toLowerCase();
  const [r, g, b] = [s[1], s[2], s[3]];
  return `#${r}${r}${g}${g}${b}${b}`.toLowerCase();
}

// ─── Palette presets (mirror reporting/config.py ColorScheme.from_palette) ──

export interface PalettePreset {
  id: 'sage' | 'corporate' | 'warm' | 'monochrome';
  label: string;
  primary: string;
  secondary: string;
  accent: string;
  palette: string[];
}

export const PALETTE_PRESETS: PalettePreset[] = [
  // SAGE = reporting design-token defaults (primary/primary_dark/accent/accent_dark).
  { id: 'sage', label: 'Sage', primary: '#8fa86a', secondary: '#6d8a4a', accent: '#6a8fa8',
    palette: ['#8fa86a', '#6a8fa8', '#6d8a4a', '#4a6d8a'] },
  { id: 'corporate', label: 'Corporate', primary: '#4a6fa5', secondary: '#3a5a8a', accent: '#5a8fa5',
    palette: ['#4a6fa5', '#5a8fa5', '#3a5a8a', '#4a7a8a'] },
  { id: 'warm', label: 'Warm', primary: '#b87c4c', secondary: '#9a6a3a', accent: '#8fa86a',
    palette: ['#b87c4c', '#8fa86a', '#9a6a3a', '#7a8a5a'] },
  { id: 'monochrome', label: 'Monochrome', primary: '#555555', secondary: '#333333', accent: '#777777',
    palette: ['#555555', '#777777', '#999999', '#333333'] },
];

export function presetToBranding(p: PalettePreset): Branding {
  return {
    colors: { primary: p.primary, secondary: p.secondary, accent: p.accent, palette: [...p.palette] },
    source: 'manual',
    confirmed: true,
  };
}

/** Which preset (if any) a primary hex corresponds to — used to highlight the
 * active row and to round-trip the global default back into the select. */
export function matchPresetId(primary: string | null | undefined): PalettePreset['id'] | null {
  if (!primary) return null;
  const p = PALETTE_PRESETS.find(x => x.primary.toLowerCase() === primary.trim().toLowerCase());
  return p ? p.id : null;
}

/** Swatch colors implied by a branding dict (palette, else primary/secondary/accent). */
export function brandSwatches(b: Branding | null | undefined): string[] {
  const c = b?.colors ?? {};
  const palette = (c.palette ?? []).filter(Boolean);
  if (palette.length > 0) return palette;
  return [c.primary, c.secondary, c.accent].filter((x): x is string => !!x);
}

export const MAX_PALETTE = 10;

export const FONT_SUGGESTIONS = [
  'Inter', 'Helvetica Neue', 'Arial', 'Roboto', 'Open Sans', 'Lato',
  'Montserrat', 'Source Sans Pro', 'Georgia', 'Garamond', 'Times New Roman', 'Futura',
];
