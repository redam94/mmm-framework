"""Client-side engine for the interactive MMM Results Report.

One raw-string JavaScript module, injected verbatim after the JSON payload
(``window.__IR_DATA__``) and theme (``window.__IR_THEME__``). It only ever
*re-aggregates* the embedded posterior draws (window sums, quantiles,
grid interpolation) — the statistical semantics are fixed in
:mod:`mmm_framework.reporting.interactive.facts`.
"""

from __future__ import annotations

__all__ = ["INTERACTIVE_REPORT_JS"]

INTERACTIVE_REPORT_JS = r"""
(function () {
  'use strict';
  var IR = window.__IR_DATA__;
  var TH = window.__IR_THEME__ || {};
  if (!IR) return;

  // ── decoding + math ────────────────────────────────────────────────────
  function b64f32(b64) {
    var bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  function Mat(b64, rows, cols) {
    this.a = b64f32(b64); this.rows = rows; this.cols = cols;
  }
  Mat.prototype.rowSumWindow = function (s, e) {
    var out = new Float64Array(this.rows);
    for (var d = 0; d < this.rows; d++) {
      var base = d * this.cols, acc = 0;
      for (var t = s; t <= e; t++) acc += this.a[base + t];
      out[d] = acc;
    }
    return out;
  };
  Mat.prototype.col = function (j) {
    var out = new Float64Array(this.rows);
    for (var d = 0; d < this.rows; d++) out[d] = this.a[d * this.cols + j];
    return out;
  };
  function quantile(sorted, q) {
    var n = sorted.length;
    if (!n) return NaN;
    var pos = q * (n - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function summar(draws, interval) {
    var v = Array.prototype.slice.call(draws).filter(isFinite);
    if (!v.length) return null;
    v.sort(function (a, b) { return a - b; });
    var alpha = (1 - (interval || IR.meta.interval || 0.9)) / 2;
    var mean = 0;
    for (var i = 0; i < v.length; i++) mean += v[i];
    mean /= v.length;
    return {
      mean: mean,
      lower: quantile(v, alpha), upper: quantile(v, 1 - alpha),
      lower50: quantile(v, 0.25), upper50: quantile(v, 0.75)
    };
  }
  function addVec(a, b) { for (var i = 0; i < a.length; i++) a[i] += b[i]; return a; }
  function sumRange(arr, s, e) {
    var acc = 0;
    for (var t = s; t <= e; t++) { var v = arr[t]; if (v != null && isFinite(v)) acc += v; }
    return acc;
  }

  // ── formatting ─────────────────────────────────────────────────────────
  function fmt(v, d) {
    if (v == null || !isFinite(v)) return '—';
    var av = Math.abs(v);
    if (av >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (av >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (av >= 1e4) return (v / 1e3).toFixed(0) + 'K';
    if (av >= 100) return v.toFixed(d == null ? 0 : d);
    return v.toFixed(d == null ? 2 : d);
  }
  function fmtPct(v) { return (v == null || !isFinite(v)) ? '—' : (v * 100).toFixed(0) + '%'; }
  function ciTxt(s, d) { return s ? fmt(s.lower, d) + ' – ' + fmt(s.upper, d) : '—'; }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── colors / layout ────────────────────────────────────────────────────
  function rgba(color, a) {
    if (!color) return 'rgba(90,107,90,' + a + ')';
    if (color[0] === '#') {
      var h = color.slice(1);
      if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
      var n = parseInt(h, 16);
      return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
    }
    if (color.indexOf('hsl(') === 0) return color.replace('hsl(', 'hsla(').replace(')', ',' + a + ')');
    return color;
  }
  function chColor(ch) { return (TH.channels || {})[ch] || TH.accent || '#5a7a52'; }
  var CFG = { displayModeBar: false, responsive: true };
  function baseLayout(extra) {
    var base = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: TH.font || 'sans-serif', size: 12, color: TH.ink || '#3a4838' },
      margin: { l: 60, r: 20, t: 24, b: 44 },
      xaxis: { gridcolor: TH.grid || '#e9e5d4', zeroline: false },
      yaxis: { gridcolor: TH.grid || '#e9e5d4', zeroline: false },
      hovermode: 'closest', showlegend: false
    };
    if (extra) {
      for (var k in extra) {
        if (k === 'xaxis' || k === 'yaxis') {
          for (var k2 in extra[k]) base[k][k2] = extra[k][k2];
        } else base[k] = extra[k];
      }
    }
    return base;
  }

  // ── shared data ────────────────────────────────────────────────────────
  var P = IR.periods.length;
  var CH = IR.meta.channels;
  var D = IR.contrib.n_draws;
  var contrib = {}, marginal = {};
  CH.forEach(function (ch) {
    contrib[ch] = new Mat(IR.contrib.draws_b64[ch], D, P);
    marginal[ch] = new Mat(IR.marginal.draws_b64[ch], D, P);
  });
  var curveM = {}, LV = IR.curves.multipliers, DC = IR.curves.n_draws;
  CH.forEach(function (ch) { curveM[ch] = new Mat(IR.curves.draws_b64[ch], DC, LV.length); });
  var meta = IR.divisor_meta || {};
  function isMon(ch) { return !!(meta[ch] || {}).is_monetary; }
  function refOf(ch) { var r = (meta[ch] || {}).reference; return (r == null) ? 1.0 : r; }
  var monetary = CH.filter(isMon), efficiency = CH.filter(function (c) { return !isMon(c); });
  var ivPct = Math.round((IR.meta.interval || 0.9) * 100);

  function windowSpend(ch, s, e) { return sumRange(IR.spend[ch], s, e); }
  function roiDraws(ch, s, e) {
    var den = windowSpend(ch, s, e);
    if (den <= 0) return null;
    var num = contrib[ch].rowSumWindow(s, e);
    for (var d = 0; d < num.length; d++) num[d] /= den;
    return num;
  }
  function mroasDraws(ch, s, e) {
    var den = windowSpend(ch, s, e) * (IR.marginal.bump_pct / 100);
    if (den <= 0) return null;
    var num = marginal[ch].rowSumWindow(s, e);
    for (var d = 0; d < num.length; d++) num[d] /= den;
    return num;
  }

  // ── window control component ───────────────────────────────────────────
  function windowControl(rootId, onChange) {
    var root = document.getElementById(rootId);
    if (!root) return { s: 0, e: P - 1 };
    var st = { s: 0, e: P - 1 };
    var startSel = document.createElement('select'), endSel = document.createElement('select');
    startSel.className = 'ir-select'; endSel.className = 'ir-select';
    IR.periods.forEach(function (p, i) {
      startSel.add(new Option(p, i)); endSel.add(new Option(p, i));
    });
    startSel.value = 0; endSel.value = P - 1;
    function apply(s, e) {
      if (s > e) { var t = s; s = e; e = t; }
      st.s = s; st.e = e; startSel.value = s; endSel.value = e;
      onChange(st.s, st.e);
    }
    startSel.onchange = function () { apply(+startSel.value, +endSel.value); };
    endSel.onchange = function () { apply(+startSel.value, +endSel.value); };
    var lbl1 = document.createElement('span'); lbl1.className = 'ir-lbl'; lbl1.textContent = 'From';
    var lbl2 = document.createElement('span'); lbl2.className = 'ir-lbl'; lbl2.textContent = 'to';
    root.appendChild(lbl1); root.appendChild(startSel);
    root.appendChild(lbl2); root.appendChild(endSel);
    var presets = [['Full', 0, P - 1]];
    if (P > 52) presets.push(['Last 52w', P - 52, P - 1]);
    if (P > 26) presets.push(['Last 26w', P - 26, P - 1]);
    var years = {};
    IR.periods.forEach(function (p, i) {
      var y = p.slice(0, 4);
      if (!years[y]) years[y] = [i, i]; else years[y][1] = i;
    });
    Object.keys(years).sort().slice(-3).forEach(function (y) {
      if (years[y][1] - years[y][0] >= 12) presets.push([y, years[y][0], years[y][1]]);
    });
    presets.forEach(function (pr) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'ir-btn'; b.textContent = pr[0];
      b.onclick = function () { apply(pr[1], pr[2]); };
      root.appendChild(b);
    });
    return st;
  }

  function kpiCard(label, value, ci) {
    return '<div class="kpi"><div class="label">' + esc(label) + '</div>' +
      '<div class="value">' + value + '</div>' +
      '<div class="ci">' + ci + '</div></div>';
  }

  // ── executive summary ──────────────────────────────────────────────────
  function renderExec(s, e) {
    var el = document.getElementById('execCards');
    if (!el) return;
    var totalKpi = sumRange(IR.actual_national, s, e);
    var media = new Float64Array(D);
    CH.forEach(function (ch) { addVec(media, contrib[ch].rowSumWindow(s, e)); });
    var mSum = summar(media);
    var shareSum = null;
    if (totalKpi > 0) {
      var sh = new Float64Array(D);
      for (var d = 0; d < D; d++) sh[d] = media[d] / totalKpi;
      shareSum = summar(sh);
    }
    var blended = null, spendTot = 0;
    if (monetary.length) {
      var num = new Float64Array(D);
      monetary.forEach(function (ch) {
        addVec(num, contrib[ch].rowSumWindow(s, e));
        spendTot += windowSpend(ch, s, e);
      });
      if (spendTot > 0) {
        for (var d2 = 0; d2 < D; d2++) num[d2] /= spendTot;
        blended = summar(num);
      }
    }
    var kpiName = IR.meta.kpi || 'KPI';
    var html = kpiCard('Total ' + kpiName, fmt(totalKpi),
      IR.periods[s] + ' → ' + IR.periods[e]);
    html += kpiCard('Attributed to media', mSum ? fmt(mSum.mean) : '—',
      ivPct + '% CI: ' + ciTxt(mSum));
    if (shareSum) {
      html += kpiCard('Share of ' + kpiName + ' from media', fmtPct(shareSum.mean),
        ivPct + '% CI: ' + fmtPct(shareSum.lower) + ' – ' + fmtPct(shareSum.upper));
    }
    if (blended) {
      html += kpiCard('Blended media ROI', blended.mean.toFixed(2),
        ivPct + '% CI: ' + blended.lower.toFixed(2) + ' – ' + blended.upper.toFixed(2));
    } else if (efficiency.length && !monetary.length) {
      html += kpiCard('Blended media ROI', '—',
        'not defined: channels are measured in volume, not spend');
    }
    el.innerHTML = html;
  }

  // ── model fit ──────────────────────────────────────────────────────────
  var BAND_ALPHA = { '95': 0.10, '90': 0.16, '80': 0.24, '50': 0.34 };
  function renderFit(geo) {
    var srs = IR.fit.series[geo];
    if (!srs) return;
    var x = IR.periods, traces = [], acc = TH.accent || '#5a7a52';
    ['95', '90', '80', '50'].forEach(function (lvl) {
      var b = srs.bands[lvl];
      if (!b) return;
      traces.push({ x: x, y: b.lo, type: 'scatter', mode: 'lines',
        line: { width: 0 }, hoverinfo: 'skip', showlegend: false, connectgaps: false });
      traces.push({ x: x, y: b.hi, type: 'scatter', mode: 'lines',
        line: { width: 0 }, fill: 'tonexty', fillcolor: rgba(acc, BAND_ALPHA[lvl]),
        name: lvl + '% interval', hoverinfo: 'skip', showlegend: false, connectgaps: false });
    });
    traces.push({ x: x, y: srs.mean, type: 'scatter', mode: 'lines',
      name: 'Posterior prediction', line: { color: acc, width: 2 }, connectgaps: false });
    traces.push({ x: x, y: srs.actual, type: 'scatter', mode: 'lines+markers',
      name: 'Observed', line: { color: TH.ink || '#2a3528', width: 1.4 },
      marker: { size: 4 }, connectgaps: false });
    var ly = baseLayout({
      showlegend: true,
      legend: { orientation: 'h', y: 1.08, x: 0, font: { size: 11 } },
      yaxis: { title: { text: IR.meta.kpi || 'KPI', font: { size: 11 } } },
      margin: { l: 64, r: 12, t: 30, b: 40 }, height: 380
    });
    Plotly.react('fitChart', traces, ly, CFG);

    var st = srs.stats || {}, el = document.getElementById('fitStats');
    if (el) {
      el.innerHTML =
        kpiCard('R²', st.r2 == null ? '—' : st.r2.toFixed(2), 'in-sample, ' + esc(geo)) +
        kpiCard('MAPE', st.mape == null ? '—' : st.mape.toFixed(1) + '%', 'mean abs. % error') +
        kpiCard('RMSE', fmt(st.rmse), 'per period') +
        kpiCard('90% band coverage', st.coverage90 == null ? '—' : fmtPct(st.coverage90),
          'well calibrated ≈ 90%');
    }
  }

  // ── forest plots ───────────────────────────────────────────────────────
  function forestTraces(rows, color) {
    // rows: [{label, sum, color?}] bottom-to-top
    var y = rows.map(function (r) { return r.label; });
    var mean = rows.map(function (r) { return r.sum ? r.sum.mean : null; });
    var t90 = {
      x: mean, y: y, type: 'scatter', mode: 'markers',
      marker: { size: 1, color: 'rgba(0,0,0,0)' },
      error_x: {
        type: 'data', symmetric: false,
        array: rows.map(function (r) { return r.sum ? r.sum.upper - r.sum.mean : 0; }),
        arrayminus: rows.map(function (r) { return r.sum ? r.sum.mean - r.sum.lower : 0; }),
        thickness: 1.4, width: 0,
        color: rows.map ? undefined : undefined
      },
      hoverinfo: 'skip', showlegend: false
    };
    var t50 = {
      x: mean, y: y, type: 'scatter', mode: 'markers',
      marker: { size: 1, color: 'rgba(0,0,0,0)' },
      error_x: {
        type: 'data', symmetric: false,
        array: rows.map(function (r) { return r.sum ? r.sum.upper50 - r.sum.mean : 0; }),
        arrayminus: rows.map(function (r) { return r.sum ? r.sum.mean - r.sum.lower50 : 0; }),
        thickness: 4, width: 0
      },
      hoverinfo: 'skip', showlegend: false
    };
    var pts = {
      x: mean, y: y, type: 'scatter', mode: 'markers',
      marker: {
        size: 9, color: rows.map(function (r) { return r.color || color; }),
        line: { color: '#fff', width: 1 }
      },
      hovertemplate: '%{y}: %{x:.3g}<br>' + ivPct + '% CI %{customdata}<extra></extra>',
      customdata: rows.map(function (r) { return r.sum ? ciTxt(r.sum) : '—'; }),
      showlegend: false
    };
    // per-row colors for error bars
    t90.error_x.color = TH.muted || '#7a8a78';
    t50.error_x.color = TH.ink || '#3a4838';
    return [t90, t50, pts];
  }
  function refShape(ref, nRows) {
    return {
      type: 'line', x0: ref, x1: ref, y0: -0.5, y1: nRows - 0.5,
      line: { color: TH.rust || '#b0563f', width: 1, dash: 'dash' }
    };
  }
  function renderForest(divId, rows, xTitle, ref, height) {
    if (!rows.length) {
      document.getElementById(divId).innerHTML =
        '<p class="note">Nothing to show for this selection.</p>';
      return;
    }
    var traces = forestTraces(rows);
    var ly = baseLayout({
      xaxis: { title: { text: xTitle, font: { size: 11 } } },
      yaxis: { automargin: true, gridcolor: 'rgba(0,0,0,0)' },
      height: height || Math.max(180, 44 * rows.length + 90),
      margin: { l: 10, r: 20, t: 10, b: 44 }
    });
    if (ref != null) ly.shapes = [refShape(ref, rows.length)];
    Plotly.react(divId, traces, ly, CFG);
  }

  // ── channel ROI section ────────────────────────────────────────────────
  var roiYoY = false;
  function yearsInWindow(s, e) {
    var out = [], cur = null;
    for (var i = s; i <= e; i++) {
      var y = IR.periods[i].slice(0, 4);
      if (!cur || cur.y !== y) { cur = { y: y, s: i, e: i }; out.push(cur); }
      else cur.e = i;
    }
    return out.filter(function (w) { return w.e - w.s >= 3; });
  }
  function renderRoi(s, e) {
    var groups = [
      { chs: monetary, div: 'roiChart', label: 'ROI (KPI units per unit spend)', ref: 1.0 },
      { chs: efficiency, div: 'roiChartEff', label: 'Efficiency (KPI units per volume unit)', ref: 0.0 }
    ];
    groups.forEach(function (g) {
      var el = document.getElementById(g.div);
      if (!el) return;
      if (!g.chs.length) { el.innerHTML = ''; return; }
      var rows = [];
      if (!roiYoY) {
        g.chs.forEach(function (ch) {
          var dr = roiDraws(ch, s, e);
          if (dr) rows.push({ label: ch, sum: summar(dr), color: chColor(ch) });
        });
        rows.sort(function (a, b) { return a.sum.mean - b.sum.mean; });
      } else {
        var yrs = yearsInWindow(s, e);
        g.chs.forEach(function (ch) {
          yrs.forEach(function (w) {
            var dr = roiDraws(ch, w.s, w.e);
            if (dr) rows.push({ label: ch + ' · ' + w.y, sum: summar(dr), color: chColor(ch) });
          });
        });
      }
      renderForest(g.div, rows, g.label + ' — ' + ivPct + '% and 50% intervals', g.ref);
    });
  }

  // ── estimand explorer ──────────────────────────────────────────────────
  var ESTIMANDS = [
    { key: 'contribution_roi', label: 'Contribution ROI (average return)', ref: function (ch) { return refOf(ch); } },
    { key: 'marginal_roas', label: 'Marginal ROAS (+' + (IR.marginal.bump_pct) + '% spend)', ref: function (ch) { return refOf(ch); } },
    { key: 'contribution', label: 'Incremental contribution (KPI units)', ref: function () { return 0; } },
    { key: 'share', label: 'Share of total KPI', ref: function () { return 0; } }
  ];
  function estimandDraws(key, ch, s, e) {
    if (key === 'contribution_roi') return roiDraws(ch, s, e);
    if (key === 'marginal_roas') return mroasDraws(ch, s, e);
    if (key === 'contribution') return contrib[ch].rowSumWindow(s, e);
    if (key === 'share') {
      var tot = sumRange(IR.actual_national, s, e);
      if (tot <= 0) return null;
      var v = contrib[ch].rowSumWindow(s, e);
      for (var d = 0; d < v.length; d++) v[d] /= tot;
      return v;
    }
    return null;
  }
  var estState = { key: 'contribution_roi' };
  function renderEstimand(s, e) {
    var spec = ESTIMANDS.filter(function (t) { return t.key === estState.key; })[0];
    var rows = [], refs = {};
    CH.forEach(function (ch) {
      var dr = estimandDraws(spec.key, ch, s, e);
      if (!dr) return;
      rows.push({ label: ch, sum: summar(dr), color: chColor(ch) });
      refs[spec.ref(ch)] = true;
    });
    rows.sort(function (a, b) { return a.sum.mean - b.sum.mean; });
    var refVals = Object.keys(refs);
    var ref = refVals.length === 1 ? +refVals[0] : null;
    renderForest('estimandChart', rows, spec.label + ' — ' + ivPct + '% and 50% intervals', ref);
    var note = document.getElementById('estimandNote');
    if (note) {
      note.textContent = (spec.key === 'marginal_roas')
        ? 'Marginal ROAS: paired posterior contrast of a +' + IR.marginal.bump_pct +
          '% spend perturbation over the selected window — the return on the next dollar, not the average dollar.'
        : (spec.key === 'share')
          ? 'Per-channel incremental contribution as a share of total observed KPI in the window.'
          : '';
    }
  }

  // ── layout helper: balanced chart grids (no lonely last cell) ──────────
  function balanceGrid(el, n) {
    var cols = 3;
    if (n % 3 === 0) cols = 3;
    else if (n % 2 === 0) cols = 2;
    else if (n <= 3) cols = n;
    el.style.gridTemplateColumns = 'repeat(' + cols + ', 1fr)';
  }

  // ── response / ROI / mROI curves + per-channel deep dive ───────────────
  var curveMode = 'response';
  var curveChannel = 'all';
  var CURVE_MODES = [
    ['response', 'Response', 'Weekly contribution'],
    ['roi', 'ROI', 'ROI at spend level'],
    ['mroi', 'Marginal ROI', 'Marginal ROI']
  ];
  function curveSeries(ch, mode) {
    // Per-draw weekly-average curves over the multiplier grid.
    var spendTot = IR.curves.spend_total[ch] || 0;
    var nP = IR.curves.n_periods || P;
    var m = curveM[ch];
    var xs = [], med = [], lo = [], hi = [];
    var L = LV.length;
    for (var l = 0; l < L; l++) {
      var mult = LV[l];
      var draws = m.col(l);
      var x = mult * spendTot / nP;
      var ys = null;
      if (mode === 'response') {
        ys = draws.map(function (v) { return v / nP; });
      } else if (mode === 'roi') {
        if (mult <= 0 || spendTot <= 0) continue;
        ys = draws.map(function (v) { return v / (mult * spendTot); });
      } else { // mroi: centered finite difference on the grid
        if (l === 0 || l === L - 1 || spendTot <= 0) continue;
        var prev = m.col(l - 1), next = m.col(l + 1);
        var dx = (LV[l + 1] - LV[l - 1]) * spendTot;
        ys = new Float64Array(draws.length);
        for (var d = 0; d < draws.length; d++) ys[d] = (next[d] - prev[d]) / dx;
      }
      var sm = summar(ys);
      if (!sm) continue;
      xs.push(x); med.push(sm.mean); lo.push(sm.lower); hi.push(sm.upper);
    }
    return { xs: xs, med: med, lo: lo, hi: hi, avgWeekly: spendTot / nP };
  }
  function curveChart(divId, ch, mode, height, titleTxt) {
    var cs = curveSeries(ch, mode);
    if (!cs.xs.length) return false;
    var c = chColor(ch);
    var yTitle = CURVE_MODES.filter(function (m) { return m[0] === mode; })[0][2];
    var traces = [
      { x: cs.xs, y: cs.lo, type: 'scatter', mode: 'lines', line: { width: 0 },
        hoverinfo: 'skip', showlegend: false },
      { x: cs.xs, y: cs.hi, type: 'scatter', mode: 'lines', line: { width: 0 },
        fill: 'tonexty', fillcolor: rgba(c, 0.16), hoverinfo: 'skip', showlegend: false },
      { x: cs.xs, y: cs.med, type: 'scatter', mode: 'lines', name: ch,
        line: { color: c, width: 2 },
        hovertemplate: 'spend/wk %{x:.3g} → %{y:.3g}<extra>' + esc(ch) + '</extra>' }
    ];
    var shapes = [];
    if (mode !== 'mroi' || (cs.avgWeekly >= cs.xs[0] && cs.avgWeekly <= cs.xs[cs.xs.length - 1])) {
      shapes.push({
        type: 'line', x0: cs.avgWeekly, x1: cs.avgWeekly, yref: 'paper', y0: 0, y1: 1,
        line: { color: TH.gold || '#b08d3f', width: 1.2, dash: 'dot' }
      });
    }
    if (mode !== 'response') {
      shapes.push({
        type: 'line', x0: cs.xs[0], x1: cs.xs[cs.xs.length - 1], y0: refOf(ch), y1: refOf(ch),
        line: { color: TH.rust || '#b0563f', width: 1, dash: 'dash' }
      });
    }
    var ly = baseLayout({
      title: { text: titleTxt, font: { size: 12 }, x: 0.02, y: 0.98 },
      height: height, margin: { l: 48, r: 8, t: 26, b: 34 },
      xaxis: { title: { text: 'avg weekly ' + ((meta[ch] || {}).divisor_units || 'spend'), font: { size: 10 } } },
      yaxis: { title: { text: yTitle, font: { size: 10 } } },
      shapes: shapes
    });
    Plotly.newPlot(divId, traces, ly, CFG);
    return true;
  }
  function headroomSummary(ch) {
    // Per-draw extra contribution available at 2x vs current spend.
    var m = curveM[ch], L = LV.length;
    var i1 = LV.indexOf(1.0), i2 = L - 1;
    if (i1 < 0) return null;
    var cur = m.col(i1), top = m.col(i2);
    var out = new Float64Array(cur.length), ok = 0;
    for (var d = 0; d < cur.length; d++) {
      if (cur[d] > 1e-9) { out[ok++] = (top[d] - cur[d]) / cur[d]; }
    }
    return ok > 10 ? summar(out.slice(0, ok)) : null;
  }
  function renderCurves() {
    var grid = document.getElementById('curvesGrid');
    if (!grid) return;
    grid.innerHTML = '';
    var modeCtl = document.getElementById('curveModeCtl');
    if (curveChannel === 'all') {
      if (modeCtl) modeCtl.style.display = '';
      var shown = 0;
      CH.forEach(function (ch, i) {
        var cell = document.createElement('div');
        cell.className = 'sat-cell';
        var div = document.createElement('div');
        div.id = 'curve_' + i; cell.appendChild(div); grid.appendChild(cell);
        if (!curveChart(div.id, ch, curveMode, 240, esc(ch))) cell.remove();
        else shown++;
      });
      balanceGrid(grid, shown);
      return;
    }
    // Deep dive: one channel, all three curve views + summary cards.
    if (modeCtl) modeCtl.style.display = 'none';
    var ch = curveChannel;
    var cards = document.createElement('div');
    cards.className = 'kpi-grid';
    var nP = IR.curves.n_periods || P;
    var units = (meta[ch] || {}).divisor_units || 'spend';
    var roi = summar(roiDraws(ch, 0, P - 1) || []);
    var mro = summar(mroasDraws(ch, 0, P - 1) || []);
    var head = headroomSummary(ch);
    cards.innerHTML =
      kpiCard('Avg weekly spend', fmt((IR.curves.spend_total[ch] || 0) / nP),
        esc(units) + ' · gold marker on the curves') +
      kpiCard('Contribution ROI', roi ? roi.mean.toFixed(2) : '—',
        ivPct + '% CI: ' + ciTxt(roi)) +
      kpiCard('Marginal ROAS', mro ? mro.mean.toFixed(2) : '—',
        ivPct + '% CI: ' + ciTxt(mro)) +
      kpiCard('Headroom to 2×', head ? fmtPct(head.mean) : '—',
        head ? 'extra contribution, CI ' + fmtPct(head.lower) + ' – ' +
          fmtPct(head.upper) : 'undefined at zero current response');
    grid.appendChild(cards);
    cards.style.gridColumn = '1 / -1';
    var shownDeep = 0;
    CURVE_MODES.forEach(function (mspec, i) {
      var cell = document.createElement('div');
      cell.className = 'sat-cell';
      var div = document.createElement('div');
      div.id = 'curvedeep_' + i; cell.appendChild(div); grid.appendChild(cell);
      if (!curveChart(div.id, ch, mspec[0], 300, esc(ch) + ' — ' + mspec[1])) {
        cell.remove();
      } else shownDeep++;
    });
    balanceGrid(grid, shownDeep);
  }

  // ── budget reallocator ─────────────────────────────────────────────────
  function interpDraws(ch, mult) {
    // Per-draw linear interpolation between grid levels (label: approximate).
    var m = curveM[ch], L = LV.length;
    if (mult <= LV[0]) return m.col(0);
    if (mult >= LV[L - 1]) return m.col(L - 1);
    var j = 1;
    while (j < L && LV[j] < mult) j++;
    var w = (mult - LV[j - 1]) / (LV[j] - LV[j - 1]);
    var a = m.col(j - 1), b = m.col(j), out = new Float64Array(a.length);
    for (var d = 0; d < a.length; d++) out[d] = a[d] * (1 - w) + b[d] * w;
    return out;
  }
  var reallocState = {};
  function renderRealloc() {
    var rowsEl = document.getElementById('reallocRows');
    var cardsEl = document.getElementById('reallocCards');
    if (!rowsEl || !cardsEl) return;
    var chs = monetary.length ? monetary : CH;
    if (!Object.keys(reallocState).length) {
      chs.forEach(function (ch) { reallocState[ch] = 1.0; });
    }
    // delta draws vs current allocation
    var delta = new Float64Array(DC), newSpend = 0, curSpend = 0;
    chs.forEach(function (ch) {
      var cur = interpDraws(ch, 1.0), alt = interpDraws(ch, reallocState[ch]);
      for (var d = 0; d < DC; d++) delta[d] += alt[d] - cur[d];
      curSpend += IR.curves.spend_total[ch] || 0;
      newSpend += (IR.curves.spend_total[ch] || 0) * reallocState[ch];
    });
    var dSum = summar(delta);
    cardsEl.innerHTML =
      kpiCard('Expected incremental ' + (IR.meta.kpi || 'KPI'),
        (dSum && dSum.mean >= 0 ? '+' : '') + fmt(dSum ? dSum.mean : null),
        ivPct + '% CI: ' + ciTxt(dSum) + ' · approximate') +
      kpiCard('Budget change', (newSpend - curSpend >= 0 ? '+' : '') + fmt(newSpend - curSpend),
        fmt(curSpend) + ' → ' + fmt(newSpend)) +
      kpiCard('Channels moved',
        String(chs.filter(function (c) { return Math.abs(reallocState[c] - 1) > 0.011; }).length),
        'of ' + chs.length + ' reallocatable');
    if (!rowsEl.dataset.built) {
      rowsEl.dataset.built = '1';
      rowsEl.innerHTML = '';
      chs.forEach(function (ch) {
        var row = document.createElement('div');
        row.className = 'ir-slider-row';
        row.innerHTML =
          '<span class="ir-slider-name"><span class="dot" style="background:' + chColor(ch) + '"></span>' +
          esc(ch) + '</span>' +
          '<input type="range" min="0" max="2" step="0.05" value="' + reallocState[ch] + '" data-ch="' + esc(ch) + '">' +
          '<span class="ir-slider-val mono" id="rv_' + esc(ch) + '"></span>';
        rowsEl.appendChild(row);
      });
      rowsEl.addEventListener('input', function (ev) {
        if (ev.target && ev.target.dataset && ev.target.dataset.ch) {
          reallocState[ev.target.dataset.ch] = +ev.target.value;
          renderRealloc();
        }
      });
      var reset = document.getElementById('reallocReset');
      if (reset) reset.onclick = function () {
        chs.forEach(function (ch) { reallocState[ch] = 1.0; });
        rowsEl.querySelectorAll('input[type=range]').forEach(function (inp) { inp.value = 1; });
        renderRealloc();
      };
      var opt = document.getElementById('reallocOptimize');
      if (opt) opt.onclick = function () { optimizeRealloc(chs); };
    }
    chs.forEach(function (ch) {
      var el = document.getElementById('rv_' + ch);
      if (el) {
        var st = IR.curves.spend_total[ch] || 0, nP = IR.curves.n_periods || P;
        el.textContent = '×' + reallocState[ch].toFixed(2) +
          ' (' + fmt(st * reallocState[ch] / nP) + '/wk)';
      }
    });
  }
  function meanAt(ch, mult) {
    var v = interpDraws(ch, mult), acc = 0;
    for (var d = 0; d < v.length; d++) acc += v[d];
    return acc / v.length;
  }
  function optimizeRealloc(chs) {
    // Greedy budget-neutral hill climb on the mean interpolated curves:
    // repeatedly move a small $ step from the lowest-marginal channel to the
    // highest-marginal channel. Approximate by construction.
    var budget = {}, tot = 0;
    chs.forEach(function (ch) { budget[ch] = IR.curves.spend_total[ch] || 0; tot += budget[ch]; });
    if (tot <= 0) return;
    var step = tot * 0.005, eps = 0.02;
    for (var it = 0; it < 400; it++) {
      var bestUp = null, bestDn = null;
      chs.forEach(function (ch) {
        var st = budget[ch];
        if (st <= 0) return;
        var mNow = reallocState[ch];
        var dm = step / st;
        var up = (mNow + dm <= 2) ? (meanAt(ch, mNow + eps) - meanAt(ch, mNow)) / (eps * st) : -Infinity;
        var dn = (mNow - dm >= 0) ? (meanAt(ch, mNow) - meanAt(ch, mNow - eps)) / (eps * st) : Infinity;
        if (bestUp === null || up > bestUp.g) bestUp = { ch: ch, g: up };
        if (bestDn === null || dn < bestDn.g) bestDn = { ch: ch, g: dn };
      });
      if (!bestUp || !bestDn || bestUp.ch === bestDn.ch) break;
      if (bestUp.g - bestDn.g < 1e-9) break;
      reallocState[bestUp.ch] += step / budget[bestUp.ch];
      reallocState[bestDn.ch] -= step / budget[bestDn.ch];
      reallocState[bestUp.ch] = Math.min(2, reallocState[bestUp.ch]);
      reallocState[bestDn.ch] = Math.max(0, reallocState[bestDn.ch]);
    }
    var rowsEl = document.getElementById('reallocRows');
    rowsEl.querySelectorAll('input[type=range]').forEach(function (inp) {
      inp.value = reallocState[inp.dataset.ch];
    });
    renderRealloc();
  }

  // ── sensitivity ────────────────────────────────────────────────────────
  function renderSensitivity() {
    var sens = IR.sensitivity;
    if (!sens || !sens.specs) return;
    [['sensChart', monetary, 1.0], ['sensChartEff', efficiency, 0.0]].forEach(function (g) {
      var el = document.getElementById(g[0]);
      if (!el) return;
      var chs = g[1];
      if (!chs.length) { el.innerHTML = ''; return; }
      var traces = chs.map(function (ch) {
        var s = sens.series[ch] || {};
        return {
          x: sens.specs, y: s.mean, type: 'scatter', mode: 'lines+markers', name: ch,
          line: { color: chColor(ch), width: 2 }, marker: { size: 7 },
          error_y: {
            type: 'data', symmetric: false,
            array: (s.mean || []).map(function (v, i) {
              return v == null || s.upper[i] == null ? 0 : s.upper[i] - v;
            }),
            arrayminus: (s.mean || []).map(function (v, i) {
              return v == null || s.lower[i] == null ? 0 : v - s.lower[i];
            }),
            thickness: 1, color: rgba(chColor(ch), 0.55)
          }
        };
      });
      var ly = baseLayout({
        showlegend: true, legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 11 } },
        xaxis: { tickangle: -25 },
        yaxis: { title: { text: g[2] === 1.0 ? 'Contribution ROI' : 'Efficiency', font: { size: 11 } } },
        height: 360, margin: { l: 54, r: 16, t: 34, b: 90 }
      });
      ly.shapes = [{
        type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: Math.min(0, g[2] - 1), y1: g[2],
        fillcolor: rgba(TH.rust || '#b0563f', 0.07), line: { width: 0 }
      }, {
        type: 'line', xref: 'paper', x0: 0, x1: 1, y0: g[2], y1: g[2],
        line: { color: TH.rust || '#b0563f', width: 1, dash: 'dash' }
      }];
      Plotly.newPlot(g[0], traces, ly, CFG);
    });
  }

  // ── decomposition: where the KPI comes from ────────────────────────────
  function meanPerPeriod(ch) {
    var m = contrib[ch], out = new Float64Array(P);
    for (var t = 0; t < P; t++) {
      var acc = 0;
      for (var d = 0; d < m.rows; d++) acc += m.a[d * m.cols + t];
      out[t] = acc / m.rows;
    }
    return out;
  }
  function renderDecomp() {
    var el = document.getElementById('decompChart');
    if (!el) return;
    var nat = (IR.fit.series || {}).National || {};
    var predMean = nat.mean || [];
    var chMeans = {}, totals = [];
    CH.forEach(function (ch) {
      chMeans[ch] = meanPerPeriod(ch);
      var s = 0;
      for (var t = 0; t < P; t++) s += chMeans[ch][t];
      totals.push([ch, s]);
    });
    totals.sort(function (a, b) { return b[1] - a[1]; });
    var baseline = new Array(P);
    for (var t = 0; t < P; t++) {
      var media = 0;
      CH.forEach(function (ch) { media += chMeans[ch][t]; });
      baseline[t] = (predMean[t] == null) ? null : predMean[t] - media;
    }
    var traces = [{
      x: IR.periods, y: baseline, type: 'scatter', mode: 'lines',
      stackgroup: 'kpi', name: 'Baseline & other',
      line: { width: 0.5, color: TH.muted || '#7a8a78' },
      fillcolor: rgba(TH.muted || '#7a8a78', 0.28),
      hovertemplate: 'Baseline & other: %{y:.3s}<extra>%{x}</extra>'
    }];
    totals.forEach(function (pair) {
      var ch = pair[0];
      traces.push({
        x: IR.periods, y: Array.prototype.slice.call(chMeans[ch]), type: 'scatter',
        mode: 'lines', stackgroup: 'kpi', name: ch,
        line: { width: 0.5, color: chColor(ch) },
        fillcolor: rgba(chColor(ch), 0.55),
        hovertemplate: esc(ch) + ': %{y:.3s}<extra>%{x}</extra>'
      });
    });
    traces.push({
      x: IR.periods, y: IR.actual_national, type: 'scatter', mode: 'lines',
      name: 'Observed', line: { color: TH.ink || '#2a3528', width: 1.4, dash: 'dot' },
      hovertemplate: 'Observed: %{y:.3s}<extra>%{x}</extra>'
    });
    var ly = baseLayout({
      showlegend: true, legend: { orientation: 'h', y: 1.1, x: 0, font: { size: 11 } },
      height: 380, margin: { l: 64, r: 12, t: 34, b: 40 },
      yaxis: { title: { text: IR.meta.kpi || 'KPI', font: { size: 11 } } }
    });
    Plotly.newPlot('decompChart', traces, ly, CFG);
  }
  function renderShares() {
    var el = document.getElementById('sharesChart');
    if (!el) return;
    var chs = monetary;
    if (chs.length < 2) { el.innerHTML = ''; return; }
    var spendTot = 0, spendBy = {};
    chs.forEach(function (ch) {
      spendBy[ch] = windowSpend(ch, 0, P - 1);
      spendTot += spendBy[ch];
    });
    if (spendTot <= 0) { el.innerHTML = ''; return; }
    var effTot = new Float64Array(D);
    var effBy = {};
    chs.forEach(function (ch) {
      effBy[ch] = contrib[ch].rowSumWindow(0, P - 1);
      addVec(effTot, effBy[ch]);
    });
    var rows = chs.map(function (ch) {
      var sh = new Float64Array(D);
      for (var d = 0; d < D; d++) sh[d] = effTot[d] > 0 ? effBy[ch][d] / effTot[d] : NaN;
      return { ch: ch, spend: spendBy[ch] / spendTot, eff: summar(sh) };
    }).filter(function (r) { return r.eff; });
    rows.sort(function (a, b) { return a.eff.mean - b.eff.mean; });
    var y = rows.map(function (r) { return r.ch; });
    var traces = [
      { x: rows.map(function (r) { return r.spend; }), y: y, type: 'bar',
        orientation: 'h', name: 'Share of spend',
        marker: { color: rgba(TH.gold || '#b8860b', 0.55) },
        hovertemplate: '%{y} spend share: %{x:.0%}<extra></extra>' },
      { x: rows.map(function (r) { return r.eff.mean; }), y: y, type: 'bar',
        orientation: 'h', name: 'Share of media effect',
        marker: { color: rows.map(function (r) { return rgba(chColor(r.ch), 0.8); }) },
        error_x: {
          type: 'data', symmetric: false,
          array: rows.map(function (r) { return r.eff.upper - r.eff.mean; }),
          arrayminus: rows.map(function (r) { return r.eff.mean - r.eff.lower; }),
          thickness: 1.2, color: TH.ink || '#3a4838'
        },
        hovertemplate: '%{y} effect share: %{x:.0%}<extra></extra>' }
    ];
    var ly = baseLayout({
      barmode: 'group', showlegend: true,
      legend: { orientation: 'h', y: 1.15, x: 0, font: { size: 11 } },
      height: Math.max(200, 56 * rows.length + 80),
      margin: { l: 10, r: 20, t: 30, b: 40 },
      xaxis: { tickformat: '.0%' },
      yaxis: { automargin: true, gridcolor: 'rgba(0,0,0,0)' }
    });
    Plotly.newPlot('sharesChart', traces, ly, CFG);
  }

  // ── predictive coverage calibration ────────────────────────────────────
  function renderCalibration() {
    var el = document.getElementById('calibChart');
    if (!el) return;
    var nat = (IR.fit.series || {}).National;
    if (!nat) return;
    var levels = ['50', '80', '90', '95'];
    var xs = [], ys = [];
    levels.forEach(function (lvl) {
      var b = nat.bands[lvl];
      if (!b) return;
      var inside = 0, n = 0;
      for (var t = 0; t < P; t++) {
        var a = nat.actual[t];
        if (a == null || b.lo[t] == null || b.hi[t] == null) continue;
        n++;
        if (a >= b.lo[t] && a <= b.hi[t]) inside++;
      }
      if (n > 3) { xs.push(+lvl / 100); ys.push(inside / n); }
    });
    if (!xs.length) return;
    var traces = [
      { x: [0.4, 1], y: [0.4, 1], type: 'scatter', mode: 'lines',
        line: { color: TH.muted || '#7a8a78', width: 1, dash: 'dash' },
        hoverinfo: 'skip', showlegend: false },
      { x: xs, y: ys, type: 'scatter', mode: 'lines+markers',
        line: { color: TH.accent || '#5a7a3a', width: 2 }, marker: { size: 9 },
        hovertemplate: 'nominal %{x:.0%} → empirical %{y:.0%}<extra></extra>',
        showlegend: false }
    ];
    var ly = baseLayout({
      height: 260, margin: { l: 54, r: 16, t: 12, b: 44 },
      xaxis: { title: { text: 'nominal interval level', font: { size: 10 } }, tickformat: '.0%', range: [0.4, 1] },
      yaxis: { title: { text: 'empirical coverage', font: { size: 10 } }, tickformat: '.0%', range: [0.35, 1.02] }
    });
    Plotly.newPlot('calibChart', traces, ly, CFG);
  }

  // ── mediation pathways (Sankey) ────────────────────────────────────────
  function renderPathways() {
    var el = document.getElementById('pathwaysChart');
    var med = IR.mediation;
    if (!el || !med || !med.links || !med.links.length) return;
    var nodeNames = [];
    function nodeIdx(name) {
      var i = nodeNames.indexOf(name);
      if (i < 0) { nodeNames.push(name); return nodeNames.length - 1; }
      return i;
    }
    // channels first, then mediators, then the outcome — stable layout
    CH.forEach(function (ch) {
      if (med.links.some(function (l) { return l.source === ch; })) nodeIdx(ch);
    });
    (med.mediators || []).forEach(nodeIdx);
    nodeIdx(med.outcome || 'KPI');
    var src = [], dst = [], val = [], lcol = [], hover = [];
    med.links.forEach(function (l) {
      src.push(nodeIdx(l.source));
      dst.push(nodeIdx(l.target));
      val.push(Math.max(Math.abs(l.mean), 1e-9));
      var c = (med.mediators || []).indexOf(l.source) >= 0
        ? (TH.gold || '#b8860b')
        : chColor(l.source);
      lcol.push(rgba(c, l.kind === 'direct' ? 0.5 : 0.32));
      hover.push(esc(l.source) + ' → ' + esc(l.target) +
        (l.kind === 'direct' ? ' (direct)' : '') + ': ' + fmt(l.mean) +
        ' (' + (med.interval ? Math.round(med.interval * 100) : 90) + '% CI ' +
        fmt(l.lower) + ' – ' + fmt(l.upper) + ')' +
        (l.mean < 0 ? ' — NEGATIVE flow, shown as magnitude' : ''));
    });
    var nodeCols = nodeNames.map(function (n) {
      if ((med.mediators || []).indexOf(n) >= 0) return TH.gold || '#b8860b';
      if (n === (med.outcome || 'KPI')) return TH.ink || '#2a3528';
      return chColor(n);
    });
    var trace = {
      type: 'sankey', orientation: 'h',
      node: {
        label: nodeNames, color: nodeCols, pad: 24, thickness: 16,
        line: { color: 'rgba(0,0,0,0)', width: 0 }
      },
      link: {
        source: src, target: dst, value: val, color: lcol,
        customdata: hover, hovertemplate: '%{customdata}<extra></extra>'
      }
    };
    var ly = baseLayout({ height: 380, margin: { l: 10, r: 10, t: 16, b: 16 } });
    Plotly.newPlot('pathwaysChart', [trace], ly, CFG);
  }

  // ── latent structure (loadings + trajectories) ─────────────────────────
  function renderLatent() {
    var lat = IR.latent;
    if (!lat) return;
    var lgEl = document.getElementById('latentLoadings');
    if (lgEl && lat.loadings && lat.loadings.length) {
      var rows = lat.loadings.slice().sort(function (a, b) { return a.mean - b.mean; });
      var y = rows.map(function (r) { return r.indicator; });
      var traces = [{
        x: rows.map(function (r) { return r.mean; }), y: y, type: 'bar',
        orientation: 'h',
        marker: {
          color: rows.map(function (r) {
            return rgba(r.mean >= 0 ? (TH.accent || '#5a7a3a') : (TH.rust || '#a04535'), 0.7);
          })
        },
        error_x: {
          type: 'data', symmetric: false,
          array: rows.map(function (r) { return r.upper == null ? 0 : r.upper - r.mean; }),
          arrayminus: rows.map(function (r) { return r.lower == null ? 0 : r.mean - r.lower; }),
          thickness: 1.2, color: TH.ink || '#3a4838'
        },
        hovertemplate: '%{y}: %{x:.2f}<extra></extra>'
      }];
      var ly = baseLayout({
        height: Math.max(180, 40 * rows.length + 70),
        margin: { l: 10, r: 20, t: 10, b: 40 },
        xaxis: { title: { text: 'loading on ' + esc(rows[0].factor), font: { size: 10 } } },
        yaxis: { automargin: true, gridcolor: 'rgba(0,0,0,0)' },
        shapes: [{ type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1,
          line: { color: TH.muted || '#7a8a78', width: 1, dash: 'dash' } }]
      });
      Plotly.newPlot('latentLoadings', traces, ly, CFG);
    }
    var grid = document.getElementById('latentGrid');
    if (grid && lat.trajectories && lat.trajectories.length) {
      balanceGrid(grid, lat.trajectories.length);
      lat.trajectories.forEach(function (tr, i) {
        var cell = document.createElement('div');
        cell.className = 'sat-cell';
        var div = document.createElement('div');
        div.id = 'latent_' + i; cell.appendChild(div); grid.appendChild(cell);
        var col = TH.accent || '#5a7a3a';
        var traces = [
          { x: IR.periods, y: tr.lower, type: 'scatter', mode: 'lines',
            line: { width: 0 }, hoverinfo: 'skip', showlegend: false, connectgaps: false },
          { x: IR.periods, y: tr.upper, type: 'scatter', mode: 'lines',
            line: { width: 0 }, fill: 'tonexty', fillcolor: rgba(col, 0.16),
            hoverinfo: 'skip', showlegend: false, connectgaps: false },
          { x: IR.periods, y: tr.median, type: 'scatter', mode: 'lines',
            line: { color: col, width: 2 }, connectgaps: false,
            hovertemplate: '%{y:.2f}<extra>%{x}</extra>' }
        ];
        var ly = baseLayout({
          title: { text: esc(tr.name), font: { size: 12 }, x: 0.02, y: 0.98 },
          height: 240, margin: { l: 44, r: 8, t: 26, b: 34 },
          yaxis: { title: { text: 'latent scale', font: { size: 9 } } }
        });
        Plotly.newPlot(div.id, traces, ly, CFG);
      });
    }
  }

  // ── posterior-predictive test statistics (static, from facts) ──────────
  function renderPpcStats() {
    var grid = document.getElementById('ppcStatsGrid');
    var stats = (IR.ppc_stats || {}).stats || [];
    if (!grid || !stats.length) return;
    balanceGrid(grid, stats.length);
    stats.forEach(function (s, i) {
      var cell = document.createElement('div');
      cell.className = 'sat-cell';
      var div = document.createElement('div');
      div.id = 'ppcstat_' + i; cell.appendChild(div); grid.appendChild(cell);
      var edges = s.hist.edges, counts = s.hist.counts;
      var mids = [], w = [];
      for (var j = 0; j < counts.length; j++) {
        mids.push((edges[j] + edges[j + 1]) / 2);
        w.push(edges[j + 1] - edges[j]);
      }
      var col = s.extreme ? (TH.rust || '#a04535') : (TH.accent || '#5a7a3a');
      var traces = [{
        x: mids, y: counts, type: 'bar', width: w,
        marker: { color: rgba(TH.accent || '#5a7a3a', 0.35),
          line: { color: rgba(TH.accent || '#5a7a3a', 0.6), width: 0.5 } },
        hovertemplate: '%{x:.3g}: %{y} replicates<extra></extra>'
      }];
      var ly = baseLayout({
        title: {
          text: esc(s.label) + ' — p = ' + s.bayes_p.toFixed(2) +
            (s.extreme ? ' ⚠' : ''),
          font: { size: 11, color: col }, x: 0.02, y: 0.98
        },
        height: 210, margin: { l: 36, r: 8, t: 26, b: 34 },
        bargap: 0.05,
        xaxis: { title: { text: esc(s.desc), font: { size: 9 } } },
        yaxis: { showticklabels: false },
        shapes: [{
          type: 'line', x0: s.observed, x1: s.observed,
          yref: 'paper', y0: 0, y1: 1,
          line: { color: col, width: 2 }
        }]
      });
      Plotly.newPlot(div.id, traces, ly, CFG);
    });
  }

  // ── LOO-PIT calibration (static, from facts) ───────────────────────────
  function renderLooPit() {
    var grid = document.getElementById('looPitGrid');
    var d = (IR.ppc_stats || {}).loo_pit;
    if (!grid || !d) return;
    balanceGrid(grid, 2);
    var col = d.calibrated ? (TH.accent || '#5a7a3a') : (TH.rust || '#a04535');
    var bandCol = rgba(TH.muted || '#7a8a78', 0.18);

    var c1 = document.createElement('div');
    c1.className = 'sat-cell';
    var d1 = document.createElement('div');
    d1.id = 'looPitHist'; c1.appendChild(d1); grid.appendChild(c1);
    var edges = d.hist.edges, counts = d.hist.counts;
    var mids = [], w = [];
    for (var j = 0; j < counts.length; j++) {
      mids.push((edges[j] + edges[j + 1]) / 2);
      w.push(edges[j + 1] - edges[j]);
    }
    var expected = d.n / counts.length;
    var traces = [
      { x: mids, y: d.band.lo, type: 'scatter', mode: 'lines', line: { width: 0 },
        hoverinfo: 'skip', showlegend: false },
      { x: mids, y: d.band.hi, type: 'scatter', mode: 'lines', line: { width: 0 },
        fill: 'tonexty', fillcolor: bandCol, hoverinfo: 'skip', showlegend: false },
      { x: mids, y: counts, type: 'bar', width: w,
        marker: { color: rgba(col, 0.35), line: { color: rgba(col, 0.6), width: 0.5 } },
        hovertemplate: 'PIT %{x:.2f}: %{y} observations<extra></extra>' }
    ];
    var ly = baseLayout({
      title: {
        text: 'PIT histogram — KS p = ' + d.ks_p.toFixed(3) + (d.calibrated ? '' : ' ⚠'),
        font: { size: 11, color: col }, x: 0.02, y: 0.98
      },
      height: 250, margin: { l: 44, r: 8, t: 26, b: 36 }, bargap: 0.05,
      xaxis: { title: { text: 'PIT value', font: { size: 10 } }, range: [0, 1] },
      yaxis: { title: { text: 'observations', font: { size: 10 } } },
      shapes: [{
        type: 'line', x0: 0, x1: 1, y0: expected, y1: expected,
        line: { color: TH.muted || '#7a8a78', width: 1, dash: 'dash' }
      }]
    });
    Plotly.newPlot(d1.id, traces, ly, CFG);

    var c2 = document.createElement('div');
    c2.className = 'sat-cell';
    var d2 = document.createElement('div');
    d2.id = 'looPitEcdf'; c2.appendChild(d2); grid.appendChild(c2);
    var e = d.ecdf;
    var t2 = [
      { x: e.z, y: e.lo, type: 'scatter', mode: 'lines', line: { width: 0 },
        hoverinfo: 'skip', showlegend: false },
      { x: e.z, y: e.hi, type: 'scatter', mode: 'lines', line: { width: 0 },
        fill: 'tonexty', fillcolor: bandCol, hoverinfo: 'skip', showlegend: false },
      { x: e.z, y: e.diff, type: 'scatter', mode: 'lines',
        line: { color: col, width: 2 },
        hovertemplate: 'PIT %{x:.2f}: ECDF − uniform = %{y:.3f}<extra></extra>' }
    ];
    var ly2 = baseLayout({
      title: { text: 'PIT ECDF − uniform', font: { size: 11, color: col }, x: 0.02, y: 0.98 },
      height: 250, margin: { l: 52, r: 8, t: 26, b: 36 },
      xaxis: { title: { text: 'PIT value', font: { size: 10 } }, range: [0, 1] },
      yaxis: { title: { text: 'ECDF difference', font: { size: 10 } } },
      shapes: [{
        type: 'line', x0: 0, x1: 1, y0: 0, y1: 0,
        line: { color: TH.muted || '#7a8a78', width: 1, dash: 'dash' }
      }]
    });
    Plotly.newPlot(d2.id, t2, ly2, CFG);
  }

  // ── carryover + prior/posterior (static, from facts) ───────────────────
  function renderCarryover() {
    var grid = document.getElementById('carryoverGrid');
    if (!grid) return;
    balanceGrid(grid, Object.keys(IR.carryover || {}).length);
    Object.keys(IR.carryover || {}).forEach(function (ch, i) {
      var c = IR.carryover[ch], col = chColor(ch);
      var cell = document.createElement('div');
      cell.className = 'sat-cell';
      var div = document.createElement('div');
      div.id = 'carry_' + i; cell.appendChild(div); grid.appendChild(cell);
      var hl = c.half_life || {};
      var traces = [
        { x: c.lags, y: c.lower, type: 'scatter', mode: 'lines', line: { width: 0 },
          hoverinfo: 'skip', showlegend: false },
        { x: c.lags, y: c.upper, type: 'scatter', mode: 'lines', line: { width: 0 },
          fill: 'tonexty', fillcolor: rgba(col, 0.16), hoverinfo: 'skip', showlegend: false },
        { x: c.lags, y: c.median, type: 'scatter', mode: 'lines+markers',
          line: { color: col, width: 2 }, marker: { size: 5 },
          hovertemplate: 'lag %{x}: weight %{y:.3f}<extra>' + esc(ch) + '</extra>' }
      ];
      var ly = baseLayout({
        title: {
          text: esc(ch) + ' — half-life ' + (hl.mean == null ? '—' : hl.mean.toFixed(1)) + 'w (' +
            (hl.lower == null ? '—' : hl.lower.toFixed(1)) + '–' +
            (hl.upper == null ? '—' : hl.upper.toFixed(1)) + ')',
          font: { size: 11 }, x: 0.02, y: 0.98
        },
        height: 220, margin: { l: 44, r: 8, t: 26, b: 34 },
        xaxis: { title: { text: 'weeks after spend', font: { size: 10 } }, dtick: 1 },
        yaxis: { title: { text: 'effect weight', font: { size: 10 } } }
      });
      Plotly.newPlot(div.id, traces, ly, CFG);
    });
  }
  function ppChart(divId, r, height, withTitle) {
    var col = chColor(r.channel), traces = [];
    if (r.prior && r.prior.density) {
      traces.push({
        x: r.grid, y: r.prior.density, type: 'scatter', mode: 'lines', name: 'Prior',
        line: { color: TH.muted || '#9aa498', width: 1.6, dash: 'dot' },
        fill: 'tozeroy', fillcolor: rgba(TH.muted || '#9aa498', 0.10)
      });
    }
    if (r.posterior && r.posterior.density) {
      traces.push({
        x: r.grid, y: r.posterior.density, type: 'scatter', mode: 'lines', name: 'Posterior',
        line: { color: col, width: 2 }, fill: 'tozeroy', fillcolor: rgba(col, 0.18)
      });
    }
    if (!traces.length) return false;
    var shapes = [{
      type: 'line', x0: r.reference, x1: r.reference, yref: 'paper', y0: 0, y1: 1,
      line: { color: TH.rust || '#b0563f', width: 1, dash: 'dash' }
    }];
    var ly = baseLayout({
      title: withTitle ? {
        text: esc(r.channel) + ' — ' + esc(r.label) +
          ' · post. ' + r.posterior.mean.toFixed(2) +
          ' (' + r.posterior.lower.toFixed(2) + '–' + r.posterior.upper.toFixed(2) + ')',
        font: { size: 11 }, x: 0.02, y: 0.98
      } : undefined,
      showlegend: true, legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
      height: height, margin: { l: 34, r: 8, t: withTitle ? 26 : 10, b: 44 },
      xaxis: { title: { text: esc(r.label), font: { size: 10 } } },
      yaxis: { showticklabels: false }, shapes: shapes
    });
    Plotly.newPlot(divId, traces, ly, CFG);
    return true;
  }
  var ppChannel = 'all';
  function renderPriorPosterior() {
    var grid = document.getElementById('ppGrid');
    if (!grid) return;
    grid.innerHTML = '';
    var rows = IR.prior_posterior.rows || [];
    if (ppChannel === 'all') {
      var shown = 0;
      rows.forEach(function (r, i) {
        var cell = document.createElement('div');
        cell.className = 'sat-cell';
        var div = document.createElement('div');
        div.id = 'pp_' + i; cell.appendChild(div); grid.appendChild(cell);
        if (!ppChart(div.id, r, 230, true)) cell.remove();
        else shown++;
      });
      balanceGrid(grid, shown);
      return;
    }
    // Deep dive: one channel — larger density + belief-update cards.
    var r = rows.filter(function (x) { return x.channel === ppChannel; })[0];
    if (!r) return;
    grid.style.gridTemplateColumns = '1fr';
    var cards = document.createElement('div');
    cards.className = 'kpi-grid';
    var post = r.posterior, prior = r.prior;
    var narrowing = null;
    if (prior && prior.sd > 1e-12 && post.sd != null) {
      narrowing = 1 - post.sd / prior.sd;
    }
    cards.innerHTML =
      kpiCard('Prior ' + esc(r.label),
        prior ? prior.mean.toFixed(2) : '—',
        prior ? ivPct + '% CI: ' + prior.lower.toFixed(2) + ' – ' +
          prior.upper.toFixed(2) : 'no prior draws') +
      kpiCard('Posterior ' + esc(r.label), post.mean.toFixed(2),
        ivPct + '% CI: ' + post.lower.toFixed(2) + ' – ' + post.upper.toFixed(2)) +
      kpiCard('Uncertainty reduction',
        narrowing == null ? '—' : fmtPct(narrowing),
        'posterior sd vs prior sd — how much the data spoke') +
      kpiCard('P(above ' + fmt(r.reference, 1) + ')',
        post.p_above == null ? '—' : fmtPct(post.p_above),
        prior && prior.p_above != null
          ? 'was ' + fmtPct(prior.p_above) + ' under the prior'
          : 'posterior probability');
    grid.appendChild(cards);
    var cell = document.createElement('div');
    cell.className = 'sat-cell';
    var div = document.createElement('div');
    div.id = 'pp_deep'; cell.appendChild(div); grid.appendChild(cell);
    ppChart(div.id, r, 340, true);
  }

  // ── year-over-year driver waterfall ────────────────────────────────────
  function yearIndices(y) {
    var out = [];
    for (var i = 0; i < P; i++) if (IR.periods[i].slice(0, 4) === y) out.push(i);
    return out;
  }
  function sumIdx(arr, idx) {
    var acc = 0;
    for (var i = 0; i < idx.length; i++) {
      var v = arr[idx[i]];
      if (v != null && isFinite(v)) acc += v;
    }
    return acc;
  }
  function contribYearDraws(ch, idx) {
    var m = contrib[ch], out = new Float64Array(m.rows);
    for (var d = 0; d < m.rows; d++) {
      var base = d * m.cols, acc = 0;
      for (var i = 0; i < idx.length; i++) acc += m.a[base + idx[i]];
      out[d] = acc;
    }
    return out;
  }
  function renderYoY() {
    var el = document.getElementById('yoyChart');
    if (!el || !IR.yoy) return;
    var ya = document.getElementById('yoyA').value;
    var yb = document.getElementById('yoyB').value;
    if (ya === yb) { el.innerHTML = '<p class="note">Pick two different years.</p>'; return; }
    var ia = yearIndices(ya), ib = yearIndices(yb);
    var totA = sumIdx(IR.actual_national, ia);
    var totB = sumIdx(IR.actual_national, ib);
    var mediaDelta = null;
    var drivers = CH.map(function (ch) {
      var d = new Float64Array(D);
      var a = contribYearDraws(ch, ia), b = contribYearDraws(ch, ib);
      for (var k = 0; k < D; k++) d[k] = b[k] - a[k];
      if (!mediaDelta) mediaDelta = new Float64Array(D);
      for (var k2 = 0; k2 < D; k2++) mediaDelta[k2] += d[k2];
      return { name: ch, sum: summar(d) };
    }).filter(function (x) { return x.sum; });
    drivers.sort(function (a, b) { return b.sum.mean - a.sum.mean; });
    var baseD = new Float64Array(D);
    for (var k = 0; k < D; k++) baseD[k] = (totB - totA) - mediaDelta[k];
    var baseSum = summar(baseD);

    var labels = [ya], vals = [totA], bases = [0],
      colors = [TH.ink || '#3a4838'], errHi = [0], errLo = [0],
      hovers = [ya + ' total: ' + fmt(totA)];
    var running = totA;
    var steps = [{ name: 'Baseline & other', sum: baseSum }].concat(drivers);
    steps.forEach(function (s) {
      labels.push(s.name);
      bases.push(running);
      vals.push(s.sum.mean);
      colors.push(s.sum.mean >= 0 ? (TH.accent || '#5a7a3a') : (TH.rust || '#a04535'));
      errHi.push(s.sum.upper - s.sum.mean);
      errLo.push(s.sum.mean - s.sum.lower);
      hovers.push(esc(s.name) + ': ' + (s.sum.mean >= 0 ? '+' : '') + fmt(s.sum.mean) +
        ' (' + ivPct + '% CI ' + fmt(s.sum.lower) + ' – ' + fmt(s.sum.upper) + ')');
      running += s.sum.mean;
    });
    labels.push(yb); vals.push(totB); bases.push(0);
    colors.push(TH.ink || '#3a4838'); errHi.push(0); errLo.push(0);
    hovers.push(yb + ' total: ' + fmt(totB));

    var shapes = [];
    var lvl = totA;
    for (var i = 1; i < labels.length - 1; i++) {
      shapes.push({
        type: 'line', x0: i - 1 + 0.4, x1: i - 0.4, y0: lvl, y1: lvl,
        line: { color: TH.muted || '#7a8a78', width: 1, dash: 'dot' }
      });
      lvl += vals[i];
    }
    shapes.push({
      type: 'line', x0: labels.length - 2 + 0.4, x1: labels.length - 1 - 0.4,
      y0: lvl, y1: lvl,
      line: { color: TH.muted || '#7a8a78', width: 1, dash: 'dot' }
    });

    var traces = [{
      x: labels, y: vals, base: bases, type: 'bar',
      marker: { color: colors.map(function (c) { return rgba(c, 0.75); }),
        line: { color: colors, width: 1 } },
      error_y: { type: 'data', symmetric: false, array: errHi, arrayminus: errLo,
        thickness: 1.2, width: 4, color: TH.ink || '#3a4838' },
      customdata: hovers,
      hovertemplate: '%{customdata}<extra></extra>'
    }];
    var ly = baseLayout({
      height: 400, margin: { l: 64, r: 16, t: 16, b: 60 },
      xaxis: { tickangle: -20 },
      yaxis: { title: { text: (IR.meta.kpi || 'KPI') + ' per year', font: { size: 11 } } },
      shapes: shapes, bargap: 0.35
    });
    Plotly.react('yoyChart', traces, ly, CFG);
    var note = document.getElementById('yoyNote');
    if (note) {
      var pct = totA !== 0 ? ' (' + (((totB - totA) / Math.abs(totA)) * 100).toFixed(1) + '%)' : '';
      var wks = ia.length !== ib.length
        ? ' Note: ' + ya + ' has ' + ia.length + ' weeks of data vs ' +
          ib.length + ' for ' + yb + '.'
        : '';
      note.textContent = ya + ' → ' + yb + ': ' + (totB - totA >= 0 ? '+' : '') +
        fmt(totB - totA) + pct + '. Media driver bars carry ' + ivPct +
        '% credible intervals; "Baseline & other" is the residual (trend, ' +
        'seasonality, controls, intercept) that closes to the observed change.' + wks;
    }
  }

  // ── boot ───────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', function () {
    windowControl('execWindow', renderExec);
    renderExec(0, P - 1);

    var geoSel = document.getElementById('fitGeoSelect');
    if (geoSel) {
      IR.fit.order.forEach(function (g) { geoSel.add(new Option(g, g)); });
      geoSel.onchange = function () { renderFit(geoSel.value); };
    }
    if (IR.fit.order.length) renderFit(IR.fit.order[0]);

    var roiWin = windowControl('roiWindow', renderRoi);
    var yoy = document.getElementById('roiYoY');
    if (yoy) yoy.onchange = function () { roiYoY = yoy.checked; renderRoi(roiWin.s, roiWin.e); };
    renderRoi(0, P - 1);

    var estWin = windowControl('estimandWindow', renderEstimand);
    var estSel = document.getElementById('estimandSelect');
    if (estSel) {
      ESTIMANDS.forEach(function (t) { estSel.add(new Option(t.label, t.key)); });
      estSel.onchange = function () { estState.key = estSel.value; renderEstimand(estWin.s, estWin.e); };
    }
    renderEstimand(0, P - 1);

    document.querySelectorAll('[data-curvemode]').forEach(function (b) {
      b.addEventListener('click', function () {
        curveMode = b.dataset.curvemode;
        document.querySelectorAll('[data-curvemode]').forEach(function (x) {
          x.classList.toggle('active', x === b);
        });
        renderCurves();
      });
    });
    var curveSel = document.getElementById('curveChannelSelect');
    if (curveSel) {
      curveSel.add(new Option('All channels', 'all'));
      CH.forEach(function (ch) { curveSel.add(new Option(ch, ch)); });
      curveSel.onchange = function () { curveChannel = curveSel.value; renderCurves(); };
    }
    renderCurves();
    renderDecomp();
    renderShares();
    renderCalibration();
    renderPathways();
    renderLatent();
    renderPpcStats();
    renderLooPit();
    renderCarryover();
    var ppSel = document.getElementById('ppChannelSelect');
    if (ppSel) {
      ppSel.add(new Option('All channels', 'all'));
      (IR.prior_posterior.rows || []).forEach(function (r) {
        ppSel.add(new Option(r.channel, r.channel));
      });
      ppSel.onchange = function () { ppChannel = ppSel.value; renderPriorPosterior(); };
    }
    renderPriorPosterior();
    if (IR.yoy && document.getElementById('yoyA')) {
      var yrs = IR.yoy.years || [];
      ['yoyA', 'yoyB'].forEach(function (id) {
        var sel = document.getElementById(id);
        yrs.forEach(function (y) { sel.add(new Option(y, y)); });
        sel.onchange = renderYoY;
      });
      document.getElementById('yoyA').value = yrs[yrs.length - 2];
      document.getElementById('yoyB').value = yrs[yrs.length - 1];
      renderYoY();
    }
    renderRealloc();
    renderSensitivity();
  });
})();
"""
