/**
 * App identity — single source of truth for the product name, the page-name
 * system, and browser-tab titles. Sidebar, Header, AppShell (document.title),
 * and the Login page all read from here; rename once, it changes everywhere.
 *
 * The app is **Augur**: a Roman augur read signs to decide whether an
 * undertaking had divine favor BEFORE acting — which is exactly what this
 * product does with causal evidence instead of birds. The page names keep to
 * that lexicon, each chosen to fit the page's actual job:
 *
 * - **Orrery** (/program) — a working model of cycles in motion: the T₀–T₅
 *   measurement loop at a glance.
 * - **Auspices** (/experiments) — literally "bird-watching": the formal act
 *   of taking the omens, i.e. running the test before committing the budget.
 * - **Chronicle** (/performance) — the cycle-over-cycle record of how
 *   measurement sharpened and decisions improved.
 * - **Oracle** (/workspace) — where you ask questions and get answers: the
 *   chat-aided modeling workspace.
 * - **Codex** (/knowledge) — the bound reference: reports and grounding docs.
 * - **College** (/team) — the College of Augurs: the people who practice.
 */

export const APP_NAME = 'Augur';
export const APP_TAGLINE = 'Causal measurement';

export interface PageIdentity {
  path: string;
  /** the creative name shown in nav, header, and the browser tab */
  name: string;
  /** plain-language function, shown under the nav label and beside the header title */
  hint: string;
}

export const PAGES: PageIdentity[] = [
  { path: '/program', name: 'Orrery', hint: 'The measurement cycle, in motion' },
  { path: '/experiments', name: 'Auspices', hint: 'Design · run · calibrate tests' },
  { path: '/performance', name: 'Chronicle', hint: 'Cycle-over-cycle record' },
  { path: '/workspace', name: 'Oracle', hint: 'Chat-aided modeling' },
  { path: '/knowledge', name: 'Codex', hint: 'Reports & reference docs' },
  { path: '/team', name: 'College', hint: 'Roster & roles' },
];

export function pageForPath(pathname: string): PageIdentity | undefined {
  return PAGES.find((p) => pathname === p.path || pathname.startsWith(p.path + '/'));
}

/** Browser-tab title: "Oracle · Augur" on a page, plain "Augur" elsewhere. */
export function documentTitle(pathname: string): string {
  const page = pageForPath(pathname);
  return page ? `${page.name} · ${APP_NAME}` : APP_NAME;
}
