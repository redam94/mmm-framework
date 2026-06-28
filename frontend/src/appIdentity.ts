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
 * - **Almanac** (/planner) — the forward calendar of when to act: budget
 *   allocation, flighting, and what-if scenarios for the coming periods.
 * - **Oracle** (/workspace) — where you ask questions and get answers: the
 *   chat-aided modeling workspace.
 * - **Codex** (/knowledge) — the bound reference: reports and grounding docs.
 * - **College** (/team) — the College of Augurs: the people who practice.
 * - **Curia** (/admin) — the senate-house: govern the org — members, roles,
 *   seats. Shown only to admins/owners.
 * - **Sanctum** (/settings) — one's private space: account, security, the
 *   model the agent runs on, and connected data sources.
 */

export const APP_NAME = 'Augur';
export const APP_TAGLINE = 'Causal measurement';

export interface PageIdentity {
  path: string;
  /** the creative name shown in nav, header, and the browser tab */
  name: string;
  /** plain-language function, shown under the nav label and beside the header title */
  hint: string;
  /** minimum org role to show this in nav (e.g. 'admin'); undefined = everyone */
  minRole?: string;
}

export const PAGES: PageIdentity[] = [
  { path: '/program', name: 'Orrery', hint: 'The measurement cycle, in motion' },
  { path: '/experiments', name: 'Auspices', hint: 'Design · run · calibrate tests' },
  { path: '/performance', name: 'Chronicle', hint: 'Cycle-over-cycle record' },
  { path: '/planner', name: 'Almanac', hint: 'Allocate budget & plan flights' },
  { path: '/portfolio', name: 'Constellation', hint: 'Benchmark brands across the book' },
  { path: '/workspace', name: 'Oracle', hint: 'Chat-aided modeling' },
  { path: '/atelier', name: 'Atelier', hint: 'Craft, version & share bespoke models', minRole: 'analyst' },
  { path: '/knowledge', name: 'Codex', hint: 'Reports & reference docs' },
  { path: '/team', name: 'College', hint: 'Roster & roles' },
  { path: '/admin', name: 'Curia', hint: 'Members, roles & seats', minRole: 'admin' },
  { path: '/settings', name: 'Sanctum', hint: 'Account, security & connections' },
];

/** Least → most privileged org roles (mirrors auth.models.Role ordering). */
const ROLE_RANK: Record<string, number> = { viewer: 0, analyst: 1, admin: 2, owner: 3 };

/**
 * Whether a page should be shown to a principal with the given org role.
 * Unknown/absent caller role → rank -1 (hidden from gated pages). A `minRole`
 * not present in ROLE_RANK → rank 99 (hidden from everyone) — so any new
 * `minRole` value MUST be added to ROLE_RANK above, or the page disappears.
 */
export function pageVisibleToRole(page: PageIdentity, role: string | undefined): boolean {
  if (!page.minRole) return true;
  return (ROLE_RANK[role ?? ''] ?? -1) >= (ROLE_RANK[page.minRole] ?? 99);
}

export function pageForPath(pathname: string): PageIdentity | undefined {
  return PAGES.find((p) => pathname === p.path || pathname.startsWith(p.path + '/'));
}

/** Browser-tab title: "Oracle · Augur" on a page, plain "Augur" elsewhere. */
export function documentTitle(pathname: string): string {
  const page = pageForPath(pathname);
  return page ? `${page.name} · ${APP_NAME}` : APP_NAME;
}
