import { useState, useEffect } from 'react';
import {
  CurrencyDollarIcon,
  TrashIcon,
  PlusIcon,
  ArrowPathIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';
import { useModels, useModelResults } from '../../api/hooks';
import {
  useBudgetPlans,
  useCreateBudgetPlan,
  useDeleteBudgetPlan,
  type BudgetPlanInfo,
} from '../../api/hooks/useBudgetPlans';

// ── Helpers ──────────────────────────────────────────────────────────────────

function pct(v: number) {
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
}

function fmtOutcome(v: number) {
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return v.toFixed(2);
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

// ── Sub-components ────────────────────────────────────────────────────────────

interface SliderRowProps {
  channel: string;
  value: number; // fractional: 0.1 = +10%
  onChange: (v: number) => void;
}

function SliderRow({ channel, value, onChange }: SliderRowProps) {
  const displayPct = Math.round(value * 100);
  const color =
    displayPct > 0 ? 'text-emerald-600' : displayPct < 0 ? 'text-red-500' : 'text-gray-500';
  return (
    <div className="flex items-center gap-3">
      <span className="w-36 text-sm text-gray-700 truncate" title={channel}>
        {channel}
      </span>
      <input
        type="range"
        min={-100}
        max={200}
        step={5}
        value={displayPct}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="flex-1 accent-indigo-600"
      />
      <span className={`w-14 text-right text-sm font-mono font-medium ${color}`}>
        {pct(value)}
      </span>
    </div>
  );
}

interface PlanCardProps {
  plan: BudgetPlanInfo;
  onDelete: (id: string) => void;
}

function PlanCard({ plan, onDelete }: PlanCardProps) {
  const changeColor =
    plan.outcome_change >= 0 ? 'text-emerald-600' : 'text-red-500';
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 flex flex-col gap-2 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-medium text-gray-900">{plan.name}</p>
          {plan.description && (
            <p className="text-xs text-gray-500 mt-0.5">{plan.description}</p>
          )}
          <p className="text-xs text-gray-400 mt-1">{fmtDate(plan.created_at)}</p>
        </div>
        <button
          onClick={() => onDelete(plan.plan_id)}
          className="p-1 text-gray-400 hover:text-red-500 transition-colors"
          title="Delete plan"
        >
          <TrashIcon className="h-4 w-4" />
        </button>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="bg-gray-50 rounded p-2">
          <p className="text-xs text-gray-500">Baseline</p>
          <p className="text-sm font-semibold text-gray-800">
            {fmtOutcome(plan.baseline_outcome)}
          </p>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <p className="text-xs text-gray-500">Scenario</p>
          <p className={`text-sm font-semibold ${changeColor}`}>
            {fmtOutcome(plan.scenario_outcome)}
          </p>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <p className="text-xs text-gray-500">Change</p>
          <p className={`text-sm font-semibold ${changeColor}`}>
            {pct(plan.outcome_change_pct / 100)}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-1 mt-1">
        {Object.entries(plan.spend_changes).map(([ch, v]) => (
          <span
            key={ch}
            className={`text-xs px-2 py-0.5 rounded-full ${
              v >= 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'
            }`}
          >
            {ch}: {pct(v)}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export function BudgetPage() {
  const { data: modelsData } = useModels({ status: 'completed' });
  const completedModels = modelsData?.models ?? [];

  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [spendChanges, setSpendChanges] = useState<Record<string, number>>({});
  const [planName, setPlanName] = useState('');
  const [planDesc, setPlanDesc] = useState('');
  const [showSaveForm, setShowSaveForm] = useState(false);
  const [lastResult, setLastResult] = useState<{
    baseline: number;
    scenario: number;
    change: number;
    pct: number;
  } | null>(null);

  const { data: resultsData } = useModelResults(selectedModelId || undefined);
  const channelNames: string[] = resultsData?.channel_names ?? [];

  const { data: plansData, isLoading: plansLoading } = useBudgetPlans(
    selectedModelId ? { model_id: selectedModelId } : undefined
  );
  const savedPlans = plansData?.plans ?? [];

  const createPlan = useCreateBudgetPlan();
  const deletePlan = useDeleteBudgetPlan();

  // Initialise sliders when channel names load
  useEffect(() => {
    if (channelNames.length > 0) {
      setSpendChanges((prev) => {
        const next: Record<string, number> = {};
        channelNames.forEach((ch) => {
          next[ch] = prev[ch] ?? 0;
        });
        return next;
      });
    }
  }, [channelNames.join(',')]);

  // Pre-select first completed model
  useEffect(() => {
    if (!selectedModelId && completedModels.length > 0) {
      setSelectedModelId(completedModels[0].model_id);
    }
  }, [completedModels.length]);

  const handleRun = () => {
    if (!selectedModelId) return;
    createPlan.mutate(
      {
        name: `Scenario ${new Date().toLocaleTimeString()}`,
        model_id: selectedModelId,
        spend_changes: spendChanges,
      },
      {
        onSuccess: (plan) => {
          setLastResult({
            baseline: plan.baseline_outcome,
            scenario: plan.scenario_outcome,
            change: plan.outcome_change,
            pct: plan.outcome_change_pct,
          });
          setShowSaveForm(false);
          setPlanName('');
          setPlanDesc('');
        },
      }
    );
  };

  const handleSavePlan = () => {
    if (!selectedModelId || !planName.trim()) return;
    createPlan.mutate({
      name: planName.trim(),
      description: planDesc.trim() || undefined,
      model_id: selectedModelId,
      spend_changes: spendChanges,
    });
    setShowSaveForm(false);
    setPlanName('');
    setPlanDesc('');
  };

  const handleDelete = (planId: string) => {
    deletePlan.mutate(planId);
  };

  const changeColor =
    (lastResult?.pct ?? 0) >= 0 ? 'text-emerald-600' : 'text-red-500';

  if (completedModels.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-96 text-center px-8">
        <CurrencyDollarIcon className="h-16 w-16 text-gray-300 mb-4" />
        <h2 className="text-2xl font-semibold text-gray-700 mb-2">Budget Planning</h2>
        <p className="text-gray-500 max-w-md">
          No completed models found. Fit a model first, then return here to run spend
          scenarios and save named budget plans.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto py-8 px-4 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Budget Planning</h1>
        <p className="text-sm text-gray-500 mt-1">
          Adjust channel spend allocations and compare predicted outcomes.
        </p>
      </div>

      {/* Model selector */}
      <div className="bg-white border border-gray-200 rounded-lg p-4 flex items-center gap-4">
        <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
          Model
        </label>
        <select
          value={selectedModelId}
          onChange={(e) => {
            setSelectedModelId(e.target.value);
            setSpendChanges({});
            setLastResult(null);
          }}
          className="flex-1 text-sm border border-gray-300 rounded-md px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {completedModels.map((m) => (
            <option key={m.model_id} value={m.model_id}>
              {m.name}
            </option>
          ))}
        </select>
      </div>

      {/* Sliders + result side by side */}
      {channelNames.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Sliders */}
          <div className="lg:col-span-3 bg-white border border-gray-200 rounded-lg p-5 space-y-4">
            <h2 className="text-base font-semibold text-gray-800">Spend Adjustments</h2>
            <div className="space-y-3">
              {channelNames.map((ch) => (
                <SliderRow
                  key={ch}
                  channel={ch}
                  value={spendChanges[ch] ?? 0}
                  onChange={(v) => setSpendChanges((prev) => ({ ...prev, [ch]: v }))}
                />
              ))}
            </div>
            <div className="flex gap-3 pt-2">
              <button
                onClick={handleRun}
                disabled={createPlan.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-60 transition-colors"
              >
                {createPlan.isPending ? (
                  <ArrowPathIcon className="h-4 w-4 animate-spin" />
                ) : (
                  <ArrowPathIcon className="h-4 w-4" />
                )}
                Run Scenario
              </button>
              {lastResult && (
                <button
                  onClick={() => setShowSaveForm((v) => !v)}
                  className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <PlusIcon className="h-4 w-4" />
                  Save as Plan
                </button>
              )}
            </div>

            {/* Save form */}
            {showSaveForm && (
              <div className="border-t pt-4 space-y-3">
                <input
                  type="text"
                  placeholder="Plan name *"
                  value={planName}
                  onChange={(e) => setPlanName(e.target.value)}
                  className="w-full text-sm border border-gray-300 rounded-md px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <input
                  type="text"
                  placeholder="Description (optional)"
                  value={planDesc}
                  onChange={(e) => setPlanDesc(e.target.value)}
                  className="w-full text-sm border border-gray-300 rounded-md px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <button
                  onClick={handleSavePlan}
                  disabled={!planName.trim() || createPlan.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 disabled:opacity-60 transition-colors"
                >
                  <CheckCircleIcon className="h-4 w-4" />
                  Save Plan
                </button>
              </div>
            )}
          </div>

          {/* Result card */}
          <div className="lg:col-span-2 space-y-4">
            {lastResult ? (
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h2 className="text-base font-semibold text-gray-800 mb-4">
                  Scenario Result
                </h2>
                <dl className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <dt className="text-gray-500">Baseline outcome</dt>
                    <dd className="font-semibold text-gray-900">
                      {fmtOutcome(lastResult.baseline)}
                    </dd>
                  </div>
                  <div className="flex justify-between text-sm">
                    <dt className="text-gray-500">Scenario outcome</dt>
                    <dd className={`font-semibold ${changeColor}`}>
                      {fmtOutcome(lastResult.scenario)}
                    </dd>
                  </div>
                  <div className="border-t pt-3 flex justify-between text-sm">
                    <dt className="text-gray-500">Change</dt>
                    <dd className={`font-semibold ${changeColor}`}>
                      {fmtOutcome(lastResult.change)} ({pct(lastResult.pct / 100)})
                    </dd>
                  </div>
                </dl>
              </div>
            ) : (
              <div className="bg-gray-50 border border-dashed border-gray-300 rounded-lg p-5 flex items-center justify-center text-sm text-gray-400 min-h-40">
                Run a scenario to see results here
              </div>
            )}
          </div>
        </div>
      )}

      {/* Saved plans */}
      <div>
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Saved Plans</h2>
        {plansLoading ? (
          <div className="text-sm text-gray-400">Loading plans…</div>
        ) : savedPlans.length === 0 ? (
          <p className="text-sm text-gray-400">
            No saved plans for this model yet. Run a scenario and save it.
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {savedPlans.map((plan) => (
              <PlanCard key={plan.plan_id} plan={plan} onDelete={handleDelete} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default BudgetPage;
