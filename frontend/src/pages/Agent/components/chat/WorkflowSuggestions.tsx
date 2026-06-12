import { Sparkles } from 'lucide-react';
import { useExperimentRegistry } from '../../../../api/hooks/useMeasurement';
import type { DashboardData } from '../../types';

interface Suggestion {
  label: string;
  prompt: string;
}

/**
 * Next-step suggestions derived from where the session sits in the Bayesian
 * workflow (data → EDA → causal structure → priors → fit → diagnostics →
 * decisions → experiments). One click sends the prompt — the chat stays the
 * driver, these just remove the "what do I type?" friction.
 */
function deriveSuggestions(
  dd: DashboardData,
  completedExperiments: number,
  hasPriorities: boolean,
): Suggestion[] {
  const out: Suggestion[] = [];
  const hasDataset = !!dd.dataset;
  const hasSpec = !!dd.model_spec;
  const fitted = dd.model_status === 'completed';
  const hasEda = !!dd.eda && ((dd.eda.issues?.length ?? 0) > 0 || (dd.eda.outlier_actions?.length ?? 0) > 0);

  if (!hasDataset) {
    out.push(
      { label: 'Generate demo data', prompt: 'Generate a realistic synthetic MMM dataset so I can explore the workflow.' },
      { label: 'What data do I need?', prompt: 'What data do I need for an MMM, and what format should the file be in?' },
    );
    return out;
  }
  // Readouts waiting for calibration outrank everything else once data exists.
  if (completedExperiments > 0) {
    out.push({
      label: `Calibrate ${completedExperiments} readout(s) & refit`,
      prompt: 'Apply the completed experiment readouts as calibration likelihoods (apply_experiment_calibration) and refit the model.',
    });
  }
  if (!hasEda && !fitted) {
    out.push({ label: 'Check data quality', prompt: 'Run the data-quality checks (EDA) on the loaded dataset and flag anything I should fix before modeling.' });
  }
  if (!hasSpec) {
    out.push(
      { label: 'Propose a causal DAG', prompt: 'Propose a causal DAG for my channels and controls, validate identification, and configure the model from it.' },
      { label: 'Configure the model', prompt: 'Configure a model for this dataset: pick the KPI, media channels, and the controls the causal structure requires.' },
    );
  } else if (!fitted) {
    out.push(
      { label: 'Prior predictive check', prompt: 'Run the prior predictive check and tell me whether the implied KPI scale is plausible before we fit.' },
      { label: 'Fit the model', prompt: 'Fit the model, then report convergence diagnostics and the headline ROI estimates.' },
    );
  } else {
    if (!hasPriorities) {
      out.push(
        { label: 'Diagnostics & ROI', prompt: 'Show the model diagnostics and the per-channel ROI with uncertainty — what is and isn’t decision-grade?' },
        { label: 'Prioritize experiments', prompt: 'Compute the EIG/EVOI experiment priorities and tell me which channels to test next quarter and why.' },
      );
    } else {
      out.push(
        { label: 'Design the top experiment', prompt: 'Design an experiment for the top-priority channel (design_experiment_plan): randomized geo lift with matched pairs and a power analysis if we have geo data, otherwise a budget-neutral randomized flighting schedule. Then plan_experiment to pre-register it.' },
        { label: 'Optimize the budget', prompt: 'Optimize the budget on the fitted model and tell me which reallocations the posterior actually supports.' },
      );
    }
  }
  return out.slice(0, 3);
}

export function WorkflowSuggestions({
  dashboardData,
  projectId,
  disabled,
  onSelect,
}: {
  dashboardData: DashboardData;
  projectId: string | null;
  disabled: boolean;
  onSelect: (prompt: string) => void;
}) {
  const { data: experiments = [] } = useExperimentRegistry(projectId);
  const completed = experiments.filter((e) => e.status === 'completed').length;
  const hasPriorities = !!dashboardData.experiment_priorities;
  const suggestions = deriveSuggestions(dashboardData, completed, hasPriorities);
  if (suggestions.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-1.5 px-4 pb-2">
      <Sparkles size={12} className="shrink-0 text-gold-600" />
      {suggestions.map((s) => (
        <button
          key={s.label}
          onClick={() => onSelect(s.prompt)}
          disabled={disabled}
          className="rounded-full border border-sage-300 bg-sage-100/60 px-2.5 py-1 text-xs font-medium text-sage-800 transition-colors hover:bg-sage-100 disabled:opacity-40"
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}
