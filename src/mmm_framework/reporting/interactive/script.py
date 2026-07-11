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

  // ── response / ROI / mROI curves ───────────────────────────────────────
  var curveMode = 'response';
  function curveSeries(ch) {
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
      if (curveMode === 'response') {
        ys = draws.map(function (v) { return v / nP; });
      } else if (curveMode === 'roi') {
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
  function renderCurves() {
    var grid = document.getElementById('curvesGrid');
    if (!grid) return;
    grid.innerHTML = '';
    var yTitle = curveMode === 'response' ? 'Weekly contribution'
      : curveMode === 'roi' ? 'ROI at spend level' : 'Marginal ROI';
    CH.forEach(function (ch, i) {
      var cs = curveSeries(ch);
      if (!cs.xs.length) return;
      var cell = document.createElement('div');
      cell.className = 'sat-cell';
      var div = document.createElement('div');
      div.id = 'curve_' + i; cell.appendChild(div); grid.appendChild(cell);
      var c = chColor(ch);
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
      if (curveMode !== 'mroi' || (cs.avgWeekly >= cs.xs[0] && cs.avgWeekly <= cs.xs[cs.xs.length - 1])) {
        shapes.push({
          type: 'line', x0: cs.avgWeekly, x1: cs.avgWeekly, yref: 'paper', y0: 0, y1: 1,
          line: { color: TH.gold || '#b08d3f', width: 1.2, dash: 'dot' }
        });
      }
      if (curveMode !== 'response') {
        var ref = refOf(ch);
        shapes.push({
          type: 'line', x0: cs.xs[0], x1: cs.xs[cs.xs.length - 1], y0: ref, y1: ref,
          line: { color: TH.rust || '#b0563f', width: 1, dash: 'dash' }
        });
      }
      var ly = baseLayout({
        title: { text: esc(ch), font: { size: 12 }, x: 0.02, y: 0.98 },
        height: 240, margin: { l: 48, r: 8, t: 26, b: 34 },
        xaxis: { title: { text: 'avg weekly ' + ((meta[ch] || {}).divisor_units || 'spend'), font: { size: 10 } } },
        yaxis: { title: { text: yTitle, font: { size: 10 } } },
        shapes: shapes
      });
      Plotly.newPlot(div.id, traces, ly, CFG);
    });
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

  // ── carryover + prior/posterior (static, from facts) ───────────────────
  function renderCarryover() {
    var grid = document.getElementById('carryoverGrid');
    if (!grid) return;
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
  function renderPriorPosterior() {
    var grid = document.getElementById('ppGrid');
    if (!grid) return;
    (IR.prior_posterior.rows || []).forEach(function (r, i) {
      var cell = document.createElement('div');
      cell.className = 'sat-cell';
      var div = document.createElement('div');
      div.id = 'pp_' + i; cell.appendChild(div); grid.appendChild(cell);
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
      if (!traces.length) return;
      var shapes = [{
        type: 'line', x0: r.reference, x1: r.reference, yref: 'paper', y0: 0, y1: 1,
        line: { color: TH.rust || '#b0563f', width: 1, dash: 'dash' }
      }];
      var ly = baseLayout({
        title: {
          text: esc(r.channel) + ' — ' + esc(r.label) +
            ' · post. ' + r.posterior.mean.toFixed(2) +
            ' (' + r.posterior.lower.toFixed(2) + '–' + r.posterior.upper.toFixed(2) + ')',
          font: { size: 11 }, x: 0.02, y: 0.98
        },
        showlegend: true, legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
        height: 230, margin: { l: 34, r: 8, t: 26, b: 44 },
        yaxis: { showticklabels: false }, shapes: shapes
      });
      Plotly.newPlot(div.id, traces, ly, CFG);
    });
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
    renderCurves();
    renderCarryover();
    renderPriorPosterior();
    renderRealloc();
    renderSensitivity();
  });
})();
"""
