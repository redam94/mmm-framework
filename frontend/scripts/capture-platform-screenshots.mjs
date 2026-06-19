/**
 * Capture clean platform screenshots from a PRODUCTION build of the Augur UI
 * for docs/platform-overview.html.
 *
 * Why production: a `vite preview` of `npm run build` strips dev-only artifacts
 * (React Query Devtools self-disables under NODE_ENV=production, no HMR overlay,
 * minified bundle), so the captures show the real app chrome with no developer
 * cruft.
 *
 * Prereqs (see the doc PR notes):
 *   - Backend agent app on :8000  (uv run uvicorn src.mmm_framework.api.main:app)
 *   - Demo seeded                 (uv run python scripts/seed_demo_project.py --synthetic-records)
 *   - Production preview on :5173 (npm run build && npx vite preview --port 5173)
 *
 * Auth + project selection are injected via localStorage (no login typed, no LLM
 * calls): the demo project is pre-selected and each seeded session is read-only.
 *
 * Usage (from frontend/):
 *   PREVIEW_URL=http://localhost:5173 \
 *   DEMO_PROJECT_ID=3d5dec52788a47dc8f4f801422491ae9 \
 *   node scripts/capture-platform-screenshots.mjs
 */
import { chromium } from 'playwright-core';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const BASE = process.env.PREVIEW_URL || 'http://localhost:5173';
const PROJECT_ID = process.env.DEMO_PROJECT_ID || '3d5dec52788a47dc8f4f801422491ae9';
const SESSION_ID = process.env.DEMO_SESSION_ID || '';
const here = path.dirname(fileURLToPath(import.meta.url));
const OUT = process.env.OUT_DIR || path.resolve(here, '../../docs/assets/screenshots');

// One real screen: 1280x800 logical, retina-crisp at 2x.
const VIEWPORT = { width: 1280, height: 800 };

/** Each shot: route, output file, optional tab click, content anchor, settle. */
const SHOTS = [
  {
    file: 'platform-program.png',
    path: '/program',
    anchor: 'Measurement program',
    fullPage: true,
  },
  {
    file: 'platform-experiments.png',
    path: '/experiments',
    anchor: 'Experiments',
    settle: 2600, // Plotly priority matrix
  },
  {
    file: 'platform-lifecycle.png',
    path: '/experiments',
    anchor: 'Experiments',
    tab: 'Lifecycle board',
    fullPage: true,
  },
  {
    file: 'platform-performance.png',
    path: '/performance/trajectories',
    anchor: 'Performance',
    settle: 2600,
    fullPage: true,
  },
  {
    file: 'platform-portfolio.png',
    path: '/portfolio',
    anchor: 'Portfolio',
    settle: 2200,
    fullPage: true,
  },
  {
    file: 'platform-workspace.png',
    path: SESSION_ID ? `/workspace?session=${SESSION_ID}` : '/workspace',
    anchor: null, // full-bleed; just settle
    settle: 3000,
  },
  {
    file: 'platform-knowledge.png',
    path: '/knowledge',
    anchor: 'Knowledge base',
    fullPage: true,
  },
];

async function main() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 2,
  });

  // Inject auth + selected project BEFORE any app code runs, on every navigation.
  await context.addInitScript((pid) => {
    // Plain keys read directly by api/client.ts -> isAuthenticated == true.
    localStorage.setItem('mmm_api_key', 'server-managed');
    localStorage.setItem('mmm_model_name', 'claude-sonnet-4-6');
    // zustand-persist blob for the ProjectSwitcher / workspace ("ONE source of truth").
    localStorage.setItem('mmm-project', JSON.stringify({ state: { currentProjectId: pid }, version: 0 }));
  }, PROJECT_ID);

  const page = await context.newPage();
  // Surface page errors to the runner log.
  page.on('console', (m) => { if (m.type() === 'error') console.log('  [console.error]', m.text().slice(0, 200)); });

  // Optional: ONLY=platform-lifecycle.png,platform-experiments.png to re-capture a subset.
  const only = (process.env.ONLY || '').split(',').map((s) => s.trim()).filter(Boolean);
  for (const shot of SHOTS) {
    if (only.length && !only.includes(shot.file)) continue;
    const url = `${BASE}${shot.path}`;
    process.stdout.write(`→ ${shot.file}  (${shot.path}) ... `);
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
      // App shell is up once the Augur wordmark renders.
      await page.getByText('Augur', { exact: true }).first().waitFor({ timeout: 15000 }).catch(() => {});
      if (shot.anchor) {
        await page.getByText(shot.anchor, { exact: false }).first().waitFor({ timeout: 15000 }).catch(() => {});
      }
      if (shot.tab) {
        // Tab labels can carry a count badge (e.g. "Lifecycle board 12"), so match
        // on substring via a button/role-tab locator rather than exact text.
        const tab = page.locator(
          `button:has-text("${shot.tab}"), [role="tab"]:has-text("${shot.tab}")`,
        ).first();
        await tab.click({ timeout: 8000 }).catch((e) => console.log('(tab click failed)', e.message));
      }
      await page.waitForTimeout(shot.settle ?? 1500);
      // Let fonts + any charts settle.
      await page.evaluate(() => document.fonts && document.fonts.ready).catch(() => {});
      await page.screenshot({
        path: path.join(OUT, shot.file),
        fullPage: !!shot.fullPage,
        animations: 'disabled',
      });
      console.log('ok');
    } catch (e) {
      console.log('FAILED:', e.message);
    }
  }

  await browser.close();
  console.log(`\nSaved to ${OUT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
