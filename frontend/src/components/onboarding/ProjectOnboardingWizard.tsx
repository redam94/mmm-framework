import { useState } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import { AlertTriangle, Check, Users } from 'lucide-react';
import { clsx } from 'clsx';
import {
  useCreateProject,
  useOnboardProject,
  useUpdateProject,
} from '../../api/hooks/useProjects';
import { useProjectMembers, useUsers } from '../../api/hooks/useTeam';
import type {
  ProjectOnboardingRequest,
  ProjectResponse,
} from '../../api/services/projectService';
import type { TeamRole } from '../../api/services/teamService';
import { Button } from '../ui';

const STEPS = ['Project', 'Client', 'Context', 'Team'] as const;

const ROLE_OPTIONS: TeamRole[] = ['owner', 'analyst', 'viewer'];

interface FormState {
  name: string;
  description: string;
  client_name: string;
  industry: string;
  website: string;
  markets: string;
  audience: string;
  goals: string;
  kpis: string;
  channels: string;
  constraints: string;
  notes: string;
}

const EMPTY_FORM: FormState = {
  name: '',
  description: '',
  client_name: '',
  industry: '',
  website: '',
  markets: '',
  audience: '',
  goals: '',
  kpis: '',
  channels: '',
  constraints: '',
  notes: '',
};

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-ink-700">{label}</label>
      {children}
    </div>
  );
}

const INPUT_CLS =
  'w-full rounded-md border border-line-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600';

function formFromProject(project: ProjectResponse): FormState {
  const meta = project.meta ?? {};
  const str = (k: keyof FormState) => {
    const v = meta[k];
    return typeof v === 'string' ? v : '';
  };
  return {
    ...EMPTY_FORM,
    name: project.name,
    description: project.description ?? '',
    client_name: str('client_name'),
    industry: str('industry'),
    website: str('website'),
    markets: str('markets'),
    audience: str('audience'),
    goals: str('goals'),
    kpis: str('kpis'),
    channels: str('channels'),
    constraints: str('constraints'),
    notes: str('notes'),
  };
}

/**
 * 4-step project wizard: project basics → client profile → measurement
 * context → team. Without `project` it creates: POST /projects then
 * /projects/{id}/onboarding (which ingests the brief into the project
 * knowledge base). With `project` it edits in place: PATCH /projects/{id}
 * then the same onboarding POST, which merges the profile, replaces the
 * member list, and re-renders the project brief in the KB.
 */
export function ProjectOnboardingWizard({
  onClose,
  onCreated,
  project,
}: {
  onClose: () => void;
  onCreated: (projectId: string) => void;
  /** When set, the wizard edits this project's info instead of creating. */
  project?: ProjectResponse;
}) {
  const isEdit = !!project;
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<FormState>(() =>
    project ? formFromProject(project) : EMPTY_FORM,
  );
  const [members, setMembers] = useState<Record<string, TeamRole>>({});
  const [membersSeeded, setMembersSeeded] = useState(!isEdit);
  const [phase, setPhase] = useState<'form' | 'success'>('form');
  const [briefStatus, setBriefStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const create = useCreateProject();
  const update = useUpdateProject(project?.project_id ?? '');
  const onboard = useOnboardProject();
  const { data: usersData } = useUsers();
  const users = usersData?.users ?? [];

  // Edit mode: seed the member checklist from the current roster once it
  // loads (render-phase adjustment — no effect needed).
  const { data: membersData } = useProjectMembers(project?.project_id ?? null);
  if (!membersSeeded && membersData) {
    setMembers(
      Object.fromEntries(membersData.members.map((m) => [m.user_id, m.role])),
    );
    setMembersSeeded(true);
  }

  const busy = create.isPending || update.isPending || onboard.isPending;
  const set = (key: keyof FormState) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const toggleMember = (userId: string, defaultRole: TeamRole) =>
    setMembers((m) => {
      if (userId in m) {
        const next = { ...m };
        delete next[userId];
        return next;
      }
      return { ...m, [userId]: defaultRole };
    });

  const createOnly = async () => {
    if (!form.name.trim() || busy) return;
    setError(null);
    try {
      const project = await create.mutateAsync({
        name: form.name.trim(),
        description: form.description.trim() || undefined,
      });
      onCreated(project.project_id);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not create the project.');
    }
  };

  const createWithOnboarding = async () => {
    if (!form.name.trim() || busy) return;
    setError(null);
    const clean = (v: string) => v.trim() || undefined;
    const body: ProjectOnboardingRequest = {
      client_name: clean(form.client_name),
      industry: clean(form.industry),
      website: clean(form.website),
      markets: clean(form.markets),
      audience: clean(form.audience),
      goals: clean(form.goals),
      kpis: clean(form.kpis),
      channels: clean(form.channels),
      constraints: clean(form.constraints),
      notes: clean(form.notes),
      members: Object.entries(members).map(([user_id, role]) => ({ user_id, role })),
    };
    try {
      const project = await create.mutateAsync({
        name: form.name.trim(),
        description: form.description.trim() || undefined,
      });
      const res = await onboard.mutateAsync({ projectId: project.project_id, body });
      onCreated(project.project_id);
      setBriefStatus(res.brief_status);
      setPhase('success');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not create the project.');
    }
  };

  const saveEdit = async () => {
    if (!project || !form.name.trim() || busy) return;
    setError(null);
    // Send every profile field (empty string clears it in the stored meta —
    // the rendered brief skips blanks), and the full member list (replaces).
    const t = (v: string) => v.trim();
    const body: ProjectOnboardingRequest = {
      client_name: t(form.client_name),
      industry: t(form.industry),
      website: t(form.website),
      markets: t(form.markets),
      audience: t(form.audience),
      goals: t(form.goals),
      kpis: t(form.kpis),
      channels: t(form.channels),
      constraints: t(form.constraints),
      notes: t(form.notes),
      members: Object.entries(members).map(([user_id, role]) => ({ user_id, role })),
    };
    try {
      await update.mutateAsync({
        name: t(form.name),
        description: t(form.description),
      });
      const res = await onboard.mutateAsync({ projectId: project.project_id, body });
      onCreated(project.project_id);
      setBriefStatus(res.brief_status);
      setPhase('success');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the project.');
    }
  };

  // Portaled to <body>: this renders from the sticky Header (ProjectSwitcher),
  // whose backdrop-blur would otherwise clip the fixed overlay to the header.
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/40 p-4"
      onClick={(e) => e.target === e.currentTarget && !busy && onClose()}
    >
      <div className="flex max-h-[85vh] w-full max-w-lg flex-col overflow-hidden rounded-xl bg-white shadow-2xl">
        {phase === 'success' ? (
          <div className="flex flex-col items-center px-8 py-10 text-center">
            <span className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-sage-100">
              <Check className="h-6 w-6 text-sage-700" />
            </span>
            <h2 className="font-display text-lg font-semibold text-ink-900">
              {form.name.trim()} {isEdit ? 'updated' : 'is ready'}
            </h2>
            {briefStatus === 'ready' ? (
              <p className="mt-2 text-sm text-ink-600">
                Project brief {isEdit ? 'refreshed in' : 'added to'} the knowledge base ✓
              </p>
            ) : (
              <p className="mt-2 flex items-start gap-1.5 text-sm text-gold-700">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                Profile saved; brief indexing unavailable (no embedding backend)
              </p>
            )}
            <Button className="mt-6" onClick={onClose}>
              Done
            </Button>
          </div>
        ) : (
          <>
            {/* Header + progress */}
            <div className="border-b border-line-200 px-6 py-4">
              <h2 className="font-display text-lg font-semibold text-ink-900">
                {isEdit ? 'Project info' : 'New project'}
              </h2>
              <div className="mt-3 flex items-center gap-2">
                {STEPS.map((label, i) => (
                  <div key={label} className="flex items-center gap-2">
                    <span
                      className={clsx(
                        'flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors',
                        i === step
                          ? 'bg-sage-700 text-white'
                          : i < step
                            ? 'bg-sage-100 text-sage-800'
                            : 'bg-cream-200 text-ink-400',
                      )}
                    >
                      <span className="num">{i + 1}</span>
                      <span className="hidden sm:inline">{label}</span>
                    </span>
                    {i < STEPS.length - 1 && <span className="h-px w-3 bg-line-300" />}
                  </div>
                ))}
              </div>
            </div>

            {/* Step body */}
            <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4 scrollbar-thin">
              {step === 0 && (
                <>
                  <Field label="Project name">
                    <input
                      autoFocus
                      value={form.name}
                      onChange={set('name')}
                      placeholder="e.g. FY26 Media Measurement"
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="Description (optional)">
                    <textarea
                      value={form.description}
                      onChange={set('description')}
                      rows={2}
                      placeholder="What this measurement program covers"
                      className={clsx(INPUT_CLS, 'resize-none')}
                    />
                  </Field>
                </>
              )}

              {step === 1 && (
                <>
                  <p className="text-xs text-ink-400">
                    All optional — this becomes the client profile in the project brief.
                  </p>
                  <Field label="Client name">
                    <input value={form.client_name} onChange={set('client_name')} placeholder="e.g. Aurora Beverages" className={INPUT_CLS} />
                  </Field>
                  <Field label="Industry">
                    <input value={form.industry} onChange={set('industry')} placeholder="e.g. CPG — non-alcoholic beverages" className={INPUT_CLS} />
                  </Field>
                  <Field label="Website">
                    <input value={form.website} onChange={set('website')} placeholder="https://…" className={INPUT_CLS} />
                  </Field>
                  <Field label="Markets">
                    <input value={form.markets} onChange={set('markets')} placeholder="e.g. US national + top 20 DMAs" className={INPUT_CLS} />
                  </Field>
                  <Field label="Audience">
                    <input value={form.audience} onChange={set('audience')} placeholder="e.g. Health-conscious adults 25–54" className={INPUT_CLS} />
                  </Field>
                </>
              )}

              {step === 2 && (
                <>
                  <p className="text-xs text-ink-400">
                    The more context here, the better the guide and copilot ground their answers.
                  </p>
                  <Field label="Goals">
                    <textarea value={form.goals} onChange={set('goals')} rows={2} placeholder="Grow share in sparkling; defend vs private label…" className={clsx(INPUT_CLS, 'resize-none')} />
                  </Field>
                  <Field label="KPIs">
                    <textarea value={form.kpis} onChange={set('kpis')} rows={2} placeholder="Sales = weekly scanner $ sales, national…" className={clsx(INPUT_CLS, 'resize-none')} />
                  </Field>
                  <Field label="Channels">
                    <textarea value={form.channels} onChange={set('channels')} rows={2} placeholder="TV, paid social, search, retail media, OOH…" className={clsx(INPUT_CLS, 'resize-none')} />
                  </Field>
                  <Field label="Constraints">
                    <textarea value={form.constraints} onChange={set('constraints')} rows={2} placeholder="No geo holdouts in Q4; search budget committed through June…" className={clsx(INPUT_CLS, 'resize-none')} />
                  </Field>
                  <Field label="Notes">
                    <textarea value={form.notes} onChange={set('notes')} rows={2} placeholder="Anything else the team should know — past studies, data quirks…" className={clsx(INPUT_CLS, 'resize-none')} />
                  </Field>
                </>
              )}

              {step === 3 && (
                <>
                  <p className="text-xs text-ink-400">
                    Pick who's on this program and their role (used for attribution and sign-off).
                  </p>
                  {users.length === 0 ? (
                    <div className="flex flex-col items-center rounded-lg border border-dashed border-line-300 bg-cream-200/60 px-6 py-8 text-center">
                      <Users className="mb-2 h-6 w-6 text-ink-300" strokeWidth={1.5} />
                      <p className="text-sm text-ink-600">
                        Add teammates first — you can also do this later.
                      </p>
                      <Link
                        to="/team"
                        onClick={onClose}
                        className="mt-1.5 text-sm text-sage-800 underline decoration-sage-300 hover:decoration-sage-600"
                      >
                        Go to the Team page
                      </Link>
                    </div>
                  ) : (
                    <ul className="divide-y divide-line-200 rounded-lg border border-line-200">
                      {users.map((u) => {
                        const selected = u.user_id in members;
                        return (
                          <li key={u.user_id} className="flex items-center gap-3 px-3 py-2.5">
                            <input
                              type="checkbox"
                              checked={selected}
                              onChange={() => toggleMember(u.user_id, u.role)}
                              className="h-4 w-4 accent-sage-700"
                            />
                            <div className="min-w-0 flex-1">
                              <div className="truncate text-sm font-medium text-ink-900">{u.name}</div>
                              {u.email && <div className="truncate text-xs text-ink-400">{u.email}</div>}
                            </div>
                            {selected && (
                              <select
                                value={members[u.user_id]}
                                onChange={(e) =>
                                  setMembers((m) => ({ ...m, [u.user_id]: e.target.value as TeamRole }))
                                }
                                className="rounded-md border border-line-300 px-2 py-1 text-xs text-ink-700 focus:outline-none focus:ring-2 focus:ring-sage-600"
                              >
                                {ROLE_OPTIONS.map((r) => (
                                  <option key={r} value={r}>
                                    {r}
                                  </option>
                                ))}
                              </select>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </>
              )}

              {error && (
                <p className="rounded-md border border-rust-600/30 bg-rust-100 px-3 py-2 text-sm text-rust-700">
                  {error}
                </p>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between gap-3 border-t border-line-200 px-6 py-4">
              <div>
                {step === 0 && !isEdit && (
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={createOnly}
                    disabled={!form.name.trim() || busy}
                  >
                    Create without onboarding
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Button type="button" variant="ghost" onClick={onClose} disabled={busy}>
                  Cancel
                </Button>
                {step > 0 && (
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={() => setStep((s) => s - 1)}
                    disabled={busy}
                  >
                    Back
                  </Button>
                )}
                {step < STEPS.length - 1 && (
                  <Button
                    type="button"
                    variant={isEdit ? 'secondary' : 'primary'}
                    onClick={() => setStep((s) => s + 1)}
                    disabled={step === 0 && !form.name.trim()}
                  >
                    Next
                  </Button>
                )}
                {(isEdit || step === STEPS.length - 1) && (
                  <Button
                    type="button"
                    onClick={isEdit ? saveEdit : createWithOnboarding}
                    disabled={!form.name.trim() || busy}
                  >
                    {isEdit
                      ? busy ? 'Saving…' : 'Save changes'
                      : busy ? 'Creating…' : 'Create project'}
                  </Button>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body,
  );
}
