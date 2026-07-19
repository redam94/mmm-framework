/* =============================================================================
 * blogviz.js — shared, theme-aware Plotly toolkit for the Research blog figures.
 *
 * Why this exists: the math/causal guide pages each defined a private `PT`
 * helper with hard-coded light colors and never re-rendered on the theme
 * toggle, so their charts read as dark-on-dark in dark mode. This module reads
 * the live CSS design tokens (so a figure is correct in BOTH themes) and
 * re-draws every registered figure when `data-theme` flips on <html>.
 *
 * Usage on a blog page (after loading Plotly + this file):
 *
 *     BV.register(function () {
 *         const p = BV.palette();
 *         Plotly.react('myChartDiv', traces, BV.layout({ ... }), BV.config);
 *     });
 *
 * `register(fn)` runs `fn` once immediately and again on every theme change.
 * For interactive figures, also wire your control listeners to the SAME `fn`.
 * ========================================================================== */
window.BV = (function () {
    'use strict';

    // -- Design tokens -------------------------------------------------------
    function cssVar(name, fallback) {
        var v = getComputedStyle(document.documentElement).getPropertyValue(name);
        return (v && v.trim()) || fallback;
    }

    function palette() {
        return {
            primary:     cssVar('--color-primary', '#8fa86a'),
            primaryDark: cssVar('--color-primary-dark', '#6d8a4a'),
            primaryLight:cssVar('--color-primary-light', '#a8c07e'),
            accent:      cssVar('--color-accent', '#6a8fa8'),
            accentDark:  cssVar('--color-accent-dark', '#4a6d8a'),
            accentLight: cssVar('--color-accent-light', '#89b3c8'),
            warning:     cssVar('--color-warning', '#d4a86a'),
            danger:      cssVar('--color-danger', '#c97067'),
            success:     cssVar('--color-success', '#6abf8a'),
            text:        cssVar('--color-text', '#2d3a2d'),
            muted:       cssVar('--color-text-muted', '#5a6b5a'),
            border:      cssVar('--color-border', '#d4ddd4'),
            grid:        cssVar('--color-border', '#d4ddd4'),
            surface:     cssVar('--color-surface', '#ffffff'),
            bgAlt:       cssVar('--color-bg-alt', '#f0f2ed')
        };
    }

    // Categorical color cycle — distinct, colorblind-considerate ordering.
    function cycle() {
        var p = palette();
        return [p.primary, p.accent, p.warning, p.danger, p.accentDark, p.primaryDark, p.success];
    }

    // -- Colour utilities ----------------------------------------------------
    function parseColor(c) {
        if (!c) return [128, 128, 128];
        c = c.trim();
        if (c[0] === '#') {
            if (c.length === 4) {
                return [parseInt(c[1] + c[1], 16), parseInt(c[2] + c[2], 16), parseInt(c[3] + c[3], 16)];
            }
            return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
        }
        var m = c.match(/rgba?\(([^)]+)\)/);
        if (m) {
            var parts = m[1].split(',').map(function (x) { return parseFloat(x); });
            return [parts[0], parts[1], parts[2]];
        }
        return [128, 128, 128];
    }

    // Translucent fill from any token colour (for confidence bands, shading).
    function band(c, alpha) {
        var rgb = parseColor(c);
        return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + (alpha === undefined ? 0.18 : alpha) + ')';
    }

    // -- Plotly layout -------------------------------------------------------
    function axis(p, extra) {
        return Object.assign({
            gridcolor: band(p.border, 0.6),
            zerolinecolor: p.border,
            linecolor: p.border,
            tickcolor: p.border,
            tickfont: { color: p.muted },
            title: { font: { color: p.text, size: 13 } },
            automargin: true
        }, extra || {});
    }

    function layout(overrides) {
        overrides = overrides || {};
        var p = palette();
        var base = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: "'Source Sans 3', sans-serif", color: p.muted, size: 13 },
            margin: { l: 62, r: 24, t: 30, b: 54 },
            hovermode: 'closest',
            hoverlabel: {
                font: { family: "'Source Sans 3', sans-serif", color: p.text },
                bgcolor: p.surface,
                bordercolor: p.border
            },
            legend: { font: { color: p.text } }
        };
        var out = Object.assign({}, base, overrides);
        out.xaxis = axis(p, Object.assign({}, overrides.xaxis));
        out.yaxis = axis(p, Object.assign({}, overrides.yaxis));
        if (overrides.xaxis2) out.xaxis2 = axis(p, overrides.xaxis2);
        if (overrides.yaxis2) out.yaxis2 = axis(p, overrides.yaxis2);
        out.font = Object.assign({}, base.font, overrides.font);
        out.margin = Object.assign({}, base.margin, overrides.margin);
        out.legend = Object.assign({}, base.legend, overrides.legend);
        return out;
    }

    var config = { responsive: true, displayModeBar: false };

    // -- Theme-change registry ----------------------------------------------
    var drawers = [];
    function redraw() {
        drawers.forEach(function (fn) { try { fn(); } catch (e) { /* keep others alive */ } });
    }
    function register(fn) {
        drawers.push(fn);
        // Draw once the DOM (and Plotly) are ready.
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', fn);
        } else {
            fn();
        }
        return fn;
    }

    // Re-render every figure when the theme attribute flips.
    var mo = new MutationObserver(function (muts) {
        for (var i = 0; i < muts.length; i++) {
            if (muts[i].attributeName === 'data-theme') { redraw(); break; }
        }
    });
    mo.observe(document.documentElement, { attributes: true });

    // -- Slider helper: binds an <input range> to its value label + a redraw --
    // sliders: array of { id, value, fmt } ; onChange runs after every input.
    function sliders(specs, onChange) {
        specs.forEach(function (s) {
            var input = document.getElementById(s.id);
            var label = s.value ? document.getElementById(s.value) : null;
            if (!input) return;
            function update() {
                if (label) label.textContent = s.fmt ? s.fmt(parseFloat(input.value)) : input.value;
            }
            input.addEventListener('input', function () { update(); onChange(); });
            update();
        });
    }

    // -- Math helpers --------------------------------------------------------
    function linspace(a, b, n) {
        var out = new Array(n), step = (b - a) / (n - 1);
        for (var i = 0; i < n; i++) out[i] = a + i * step;
        return out;
    }
    function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

    // Deterministic PRNG so every reader sees the same "random" figure.
    function mulberry32(seed) {
        var a = seed >>> 0;
        return function () {
            a |= 0; a = (a + 0x6D2B79F5) | 0;
            var t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
    // Standard normal via Box–Muller, driven by a mulberry32 stream.
    function randn(rng) {
        var u = 0, v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    // Error function + normal CDF (Abramowitz & Stegun 7.1.26).
    function erf(x) {
        var s = x < 0 ? -1 : 1; x = Math.abs(x);
        var t = 1 / (1 + 0.3275911 * x);
        var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return s * y;
    }
    function normCdf(x, mu, sd) {
        mu = mu || 0; sd = sd === undefined ? 1 : sd;
        return 0.5 * (1 + erf((x - mu) / (sd * Math.SQRT2)));
    }
    function mean(a) { return a.reduce(function (s, x) { return s + x; }, 0) / a.length; }
    function quantile(sorted, q) {
        var pos = (sorted.length - 1) * q, base = Math.floor(pos), rest = pos - base;
        return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
    }

    return {
        cssVar: cssVar, palette: palette, cycle: cycle, band: band,
        layout: layout, config: config, register: register, redraw: redraw,
        sliders: sliders, linspace: linspace, clamp: clamp,
        mulberry32: mulberry32, randn: randn, erf: erf, normCdf: normCdf,
        mean: mean, quantile: quantile
    };
})();
