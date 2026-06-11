import { useState } from 'react';
import { ImageOff } from 'lucide-react';
import { brandSwatches, type Branding } from './lib';

/** Swatch strip + a mini mock report card so the user can see what the
 * branding will look like on generated output before confirming/saving. */
export function BrandPreview({ branding }: { branding: Branding }) {
  const swatches = brandSwatches(branding);
  const primary = branding.colors?.primary || swatches[0] || '#8fa86a';
  const accent = branding.colors?.accent || swatches[1] || '#6a8fa8';
  const headingFont = branding.fonts?.heading || undefined;
  const bodyFont = branding.fonts?.body || undefined;
  const barColors = swatches.length > 0 ? swatches : [primary, accent];

  // Track which URL failed (rather than a boolean reset in an effect) so a
  // changed logo_url automatically gets a fresh attempt.
  const [failedLogoUrl, setFailedLogoUrl] = useState<string | null>(null);
  const logoError = !!branding.logo_url && failedLogoUrl === branding.logo_url;

  return (
    <div className="space-y-3">
      {/* Swatch strip */}
      {swatches.length > 0 && (
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg overflow-hidden border border-gray-200">
            {swatches.map((c, i) => (
              <div key={`${c}-${i}`} className="w-9 h-7 flex items-end justify-center" style={{ backgroundColor: c }} />
            ))}
          </div>
          <div className="flex flex-col text-[10px] text-gray-400 font-mono leading-tight">
            {swatches.slice(0, 3).map((c, i) => <span key={`${c}-${i}`}>{c}</span>)}
          </div>
        </div>
      )}

      {/* Mini mock report card */}
      <div className="rounded-xl border border-gray-200 overflow-hidden bg-white shadow-sm">
        <div className="h-1.5" style={{ backgroundColor: primary }} />
        <div className="p-4 space-y-3">
          <div className="flex items-center gap-3">
            {branding.logo_url && !logoError ? (
              <img
                src={branding.logo_url}
                alt="logo"
                className="h-8 w-auto max-w-[120px] object-contain"
                onError={() => setFailedLogoUrl(branding.logo_url ?? null)}
              />
            ) : branding.logo_url ? (
              <span className="flex items-center justify-center h-8 w-8 rounded bg-gray-100 text-gray-300">
                <ImageOff size={14} />
              </span>
            ) : null}
            <div>
              <p className="text-sm font-bold" style={{ color: primary, fontFamily: headingFont }}>
                {branding.client_name || 'Client Name'} — Marketing Mix Review
              </p>
              <p className="text-[10px] uppercase tracking-wider" style={{ color: accent, fontFamily: headingFont }}>
                Quarterly Report
              </p>
            </div>
          </div>
          <p className="text-xs text-gray-600 leading-relaxed" style={{ fontFamily: bodyFont }}>
            Media drove an estimated 24% of revenue this quarter, led by paid search and
            connected TV. Diminishing returns suggest reallocating spend at the margin.
          </p>
          {/* A few colored bars */}
          <div className="flex items-end gap-1.5 h-12">
            {[0.9, 0.65, 0.5, 0.35, 0.25].map((h, i) => (
              <div
                key={i}
                className="flex-1 rounded-t"
                style={{ height: `${h * 100}%`, backgroundColor: barColors[i % barColors.length] }}
              />
            ))}
          </div>
          <div className="pt-2 border-t" style={{ borderColor: accent }}>
            <p className="text-[10px] text-gray-400" style={{ fontFamily: bodyFont }}>
              {branding.footer_text || 'Confidential — prepared for internal use.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
