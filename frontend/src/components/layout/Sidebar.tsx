import { useState } from 'react';
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { clsx } from 'clsx';
import {
  Bird,
  BookOpen,
  ChevronDown,
  LogOut,
  MessageSquareText,
  Orbit,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Plus,
  ScrollText,
  Telescope,
  Trash2,
  Users,
} from 'lucide-react';
import { APP_NAME, APP_TAGLINE, PAGES } from '../../appIdentity';
import { useAuthStore } from '../../stores/authStore';
import { useProjectStore } from '../../stores/projectStore';
import { useUiStore } from '../../stores/uiStore';
import {
  useCreateSession,
  useDeleteSession,
  useSessions,
  useUpdateSession,
} from '../../api/hooks/useSessions';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  hint: string;
}

// IA mirrors the measurement loop: Orrery (where the cycle stands) → Auspices
// (what to test next) → Chronicle (how measurement improved) → Oracle (the
// chat-aided way work gets done) → Codex (what the copilot grounds on) →
// College (who does it). Names + hints live in appIdentity.ts; icons match
// the augury theme (an orrery's orbits, the auspices' birds, the chronicle's
// scroll).
const NAV_ICONS: Record<string, NavItem['icon']> = {
  '/program': Orbit,
  '/experiments': Bird,
  '/performance': ScrollText,
  '/portfolio': Telescope,
  '/workspace': MessageSquareText,
  '/knowledge': BookOpen,
  '/team': Users,
};

const navigation: NavItem[] = PAGES.map((p) => ({
  name: p.name,
  href: p.path,
  icon: NAV_ICONS[p.path] ?? BookOpen,
  hint: p.hint,
}));

/** Sessions of the current project, nested under the Workspace nav item —
 * the single navigation surface for sessions (the workspace's own rail is
 * gone). Clicking one deep-links into /workspace?session=… */
function WorkspaceSessions() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const projectId = useProjectStore((s) => s.currentProjectId);
  const { data } = useSessions(
    projectId ? { project_id: projectId, limit: 25 } : { limit: 25 },
  );
  const createSession = useCreateSession();
  const updateSession = useUpdateSession();
  const deleteSession = useDeleteSession();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');

  const sessions = data?.sessions ?? [];
  const onWorkspace = location.pathname.startsWith('/workspace');
  const activeId = onWorkspace
    ? searchParams.get('session') ?? localStorage.getItem('mmm.activeThreadId')
    : null;

  const open = (id: string) => navigate(`/workspace?session=${id}`);

  const handleNew = async () => {
    try {
      const sess = await createSession.mutateAsync({
        name: `Session ${new Date().toLocaleDateString()}`,
        ...(projectId ? { project_id: projectId } : {}),
      });
      open(sess.thread_id);
    } catch (e) {
      console.error('Could not create session', e);
    }
  };

  const commitRename = (threadId: string, fallback: string) => {
    updateSession.mutate({ threadId, name: editName.trim() || fallback });
    setEditingId(null);
  };

  return (
    <ul className="ml-7 mt-0.5 max-h-72 space-y-0.5 overflow-y-auto border-l border-white/10 pl-2 scrollbar-thin">
      {sessions.map((s) => {
        const active = s.thread_id === activeId;
        const isEditing = editingId === s.thread_id;
        return (
          <li key={s.thread_id} className="group flex items-center gap-1">
            {isEditing ? (
              <input
                autoFocus
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitRename(s.thread_id, s.name);
                  if (e.key === 'Escape') setEditingId(null);
                }}
                onBlur={() => commitRename(s.thread_id, s.name)}
                className="w-full rounded border border-sage-300 bg-white/10 px-1.5 py-1 text-xs text-cream-50 focus:outline-none"
              />
            ) : (
              <button
                onClick={() => open(s.thread_id)}
                className={clsx(
                  'flex-1 truncate rounded px-2 py-1 text-left text-xs transition-colors',
                  active
                    ? 'bg-sage-600/25 font-medium text-cream-50'
                    : 'text-ink-300 hover:bg-white/5 hover:text-cream-50',
                )}
                title={s.name}
              >
                {s.name}
              </button>
            )}
            {!isEditing && (
              <span className="hidden shrink-0 items-center group-hover:flex">
                <button
                  onClick={() => {
                    setEditName(s.name);
                    setEditingId(s.thread_id);
                  }}
                  className="p-1 text-ink-300/60 hover:text-cream-50"
                  title="Rename"
                >
                  <Pencil size={10} />
                </button>
                <button
                  onClick={() => {
                    if (confirm(`Delete "${s.name}"?`)) deleteSession.mutate(s.thread_id);
                  }}
                  className="p-1 text-ink-300/60 hover:text-rust-600"
                  title="Delete"
                >
                  <Trash2 size={10} />
                </button>
              </span>
            )}
          </li>
        );
      })}
      <li>
        <button
          onClick={handleNew}
          disabled={createSession.isPending}
          className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-xs text-sage-300/90 transition-colors hover:bg-white/5 hover:text-sage-300 disabled:opacity-50"
        >
          <Plus size={11} /> New session
        </button>
      </li>
    </ul>
  );
}

export function Sidebar() {
  const location = useLocation();
  const { clearApiKey } = useAuthStore();
  const collapsed = useUiStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);
  // Sessions auto-expand while working in the workspace; manual toggle elsewhere.
  const onWorkspace = location.pathname.startsWith('/workspace');
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const showSessions = !collapsed && (onWorkspace || sessionsOpen);

  return (
    <div
      className={clsx('flex h-full flex-col', collapsed ? 'w-16' : 'w-64')}
      style={{ backgroundColor: '#2a3528' }}
    >
      {/* Wordmark */}
      <div
        className={clsx(
          'flex h-16 shrink-0 items-center gap-3',
          collapsed ? 'justify-center px-0' : 'px-6',
        )}
      >
        <img
          src="/augur.svg"
          alt=""
          className="h-8 w-8 shrink-0 rounded-md ring-1 ring-sage-300/40"
        />
        {!collapsed && (
          <div className="leading-tight">
            <span className="font-display text-lg font-semibold tracking-tight text-cream-50">
              {APP_NAME}
            </span>
            <div className="text-[10px] uppercase tracking-[0.18em] text-sage-300/80">
              {APP_TAGLINE}
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav
        className={clsx(
          'flex flex-1 flex-col overflow-y-auto py-4 scrollbar-thin',
          collapsed ? 'px-2' : 'px-3',
        )}
      >
        <ul className="flex flex-1 flex-col gap-1">
          {navigation.map((item) => {
            const isActive =
              location.pathname === item.href || location.pathname.startsWith(item.href + '/');
            const isWorkspace = item.href === '/workspace';

            return (
              <li key={item.name}>
                <div className="flex items-center">
                  <Link
                    to={item.href}
                    title={collapsed ? item.name : undefined}
                    className={clsx(
                      'group flex flex-1 items-center rounded-md text-sm font-medium transition-colors',
                      collapsed ? 'justify-center px-0 py-2.5' : 'gap-x-3 px-3 py-2.5',
                      isActive
                        ? 'bg-sage-600/25 text-cream-50'
                        : 'text-ink-300 hover:bg-white/5 hover:text-cream-50',
                    )}
                  >
                    <item.icon
                      className={clsx(
                        'h-5 w-5 shrink-0',
                        isActive ? 'text-sage-300' : 'text-ink-300 group-hover:text-sage-300',
                      )}
                      strokeWidth={1.75}
                    />
                    {!collapsed && <span className="flex-1">{item.name}</span>}
                    {!collapsed && isActive && !isWorkspace && (
                      <span className="h-1.5 w-1.5 rounded-full bg-sage-300" />
                    )}
                  </Link>
                  {/* The Workspace item expands to its sessions */}
                  {isWorkspace && !collapsed && (
                    <button
                      onClick={() => setSessionsOpen((v) => !v)}
                      className="rounded p-1.5 text-ink-300 transition-colors hover:bg-white/5 hover:text-cream-50"
                      title={showSessions ? 'Hide sessions' : 'Show sessions'}
                    >
                      <ChevronDown
                        size={14}
                        className={clsx('transition-transform', showSessions && 'rotate-180')}
                      />
                    </button>
                  )}
                </div>
                {!collapsed && (
                  <div
                    className={clsx(
                      'pl-11 text-[11px] leading-snug',
                      isActive ? 'text-sage-300/70' : 'text-ink-300/50',
                    )}
                  >
                    {item.hint}
                  </div>
                )}
                {isWorkspace && showSessions && <WorkspaceSessions />}
              </li>
            );
          })}
        </ul>

        {/* Collapse + sign out */}
        <div
          className={clsx(
            'mt-auto border-t border-white/10 pt-3',
            collapsed && 'flex flex-col items-center gap-1',
          )}
        >
          <button
            onClick={toggleSidebar}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className={clsx(
              'flex items-center rounded-md text-sm font-medium text-ink-300 transition-colors hover:bg-white/5 hover:text-cream-50',
              collapsed ? 'justify-center p-2.5' : 'w-full gap-x-3 px-3 py-2',
            )}
          >
            {collapsed ? (
              <PanelLeftOpen className="h-5 w-5 shrink-0" strokeWidth={1.75} />
            ) : (
              <>
                <PanelLeftClose className="h-5 w-5 shrink-0" strokeWidth={1.75} />
                <span>Collapse</span>
              </>
            )}
          </button>
          <button
            onClick={() => clearApiKey()}
            title={collapsed ? 'Sign out' : undefined}
            className={clsx(
              'flex items-center rounded-md text-sm font-medium text-ink-300 transition-colors hover:bg-white/5 hover:text-cream-50',
              collapsed ? 'justify-center p-2.5' : 'w-full gap-x-3 px-3 py-2',
            )}
          >
            <LogOut className="h-5 w-5 shrink-0" strokeWidth={1.75} />
            {!collapsed && <span>Sign out</span>}
          </button>
        </div>
      </nav>

      {/* Status footer */}
      {!collapsed && (
        <div className="border-t border-white/10 px-6 py-3">
          <div className="flex items-center text-xs text-ink-300/70">
            <span className="mr-2 h-2 w-2 rounded-full bg-sage-300" />
            <span>Connected</span>
          </div>
        </div>
      )}
    </div>
  );
}
