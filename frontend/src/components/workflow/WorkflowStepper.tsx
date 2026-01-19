import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';
import {
  MapIcon,
  CircleStackIcon,
  Cog6ToothIcon,
  PlayIcon,
  MagnifyingGlassIcon,
  DocumentChartBarIcon,
  CheckIcon,
  ExclamationTriangleIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { useWorkflowStore, type BayesianPhase } from '../../stores/workflowStore';

interface WorkflowPhase {
  id: BayesianPhase;
  label: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  path: string;
  description: string;
}

const WORKFLOW_PHASES: WorkflowPhase[] = [
  {
    id: 'planning',
    label: 'Planning',
    icon: MapIcon,
    path: '/planning',
    description: 'Define the generative story',
  },
  {
    id: 'data',
    label: 'Data',
    icon: CircleStackIcon,
    path: '/data',
    description: 'Upload and validate data',
  },
  {
    id: 'config',
    label: 'Configure',
    icon: Cog6ToothIcon,
    path: '/config',
    description: 'Set up model configuration',
  },
  {
    id: 'fit',
    label: 'Fit',
    icon: PlayIcon,
    path: '/fit',
    description: 'Run model fitting',
  },
  {
    id: 'diagnostics',
    label: 'Diagnostics',
    icon: MagnifyingGlassIcon,
    path: '/diagnostics',
    description: 'Check convergence',
  },
  {
    id: 'results',
    label: 'Results',
    icon: DocumentChartBarIcon,
    path: '/results',
    description: 'View and export results',
  },
];

interface WorkflowStepperProps {
  className?: string;
  showDescriptions?: boolean;
}

export function WorkflowStepper({ className, showDescriptions = false }: WorkflowStepperProps) {
  const location = useLocation();
  const { currentPhase, getPhaseCompletion } = useWorkflowStore();

  return (
    <nav className={clsx('bg-white rounded-lg shadow-sm border border-gray-200 p-4', className)}>
      <div className="flex items-center justify-between overflow-x-auto">
        {WORKFLOW_PHASES.map((phase, index) => {
          const isActive = location.pathname === phase.path ||
            location.pathname.startsWith(phase.path + '/');
          const isCurrent = phase.id === currentPhase;
          const completion = getPhaseCompletion(phase.id);
          const hasWarnings = completion.warnings.length > 0;

          return (
            <div key={phase.id} className="flex items-center">
              <Link
                to={phase.path}
                className={clsx(
                  'flex items-center gap-2 px-3 py-2 rounded-md transition-all',
                  'hover:bg-gray-100',
                  isActive && 'bg-blue-50',
                  isCurrent && !isActive && 'ring-2 ring-blue-200'
                )}
              >
                {/* Icon with status indicator */}
                <div className="relative">
                  <phase.icon
                    className={clsx(
                      'h-5 w-5',
                      isActive ? 'text-blue-600' : 'text-gray-400',
                      completion.complete && !isActive && 'text-green-500'
                    )}
                  />
                  {/* Completion badge */}
                  {completion.complete && (
                    <span className="absolute -top-1 -right-1 h-3 w-3 bg-green-500 rounded-full flex items-center justify-center">
                      <CheckIcon className="h-2 w-2 text-white" />
                    </span>
                  )}
                  {/* Warning badge */}
                  {hasWarnings && !completion.complete && (
                    <span className="absolute -top-1 -right-1 h-3 w-3 bg-yellow-500 rounded-full flex items-center justify-center">
                      <ExclamationTriangleIcon className="h-2 w-2 text-white" />
                    </span>
                  )}
                </div>

                {/* Label */}
                <div className="hidden sm:block">
                  <span
                    className={clsx(
                      'text-sm font-medium',
                      isActive ? 'text-blue-700' : 'text-gray-600',
                      completion.complete && !isActive && 'text-green-600'
                    )}
                  >
                    {phase.label}
                  </span>
                  {showDescriptions && (
                    <p className="text-xs text-gray-400">{phase.description}</p>
                  )}
                </div>

                {/* Current phase indicator */}
                {isCurrent && (
                  <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                )}
              </Link>

              {/* Connector */}
              {index < WORKFLOW_PHASES.length - 1 && (
                <ChevronRightIcon className="h-4 w-4 text-gray-300 mx-1 hidden sm:block" />
              )}
            </div>
          );
        })}
      </div>
    </nav>
  );
}

// Compact version for headers
export function WorkflowStepperCompact() {
  const { currentPhase, getPhaseCompletion, getIterationCount } = useWorkflowStore();
  const currentPhaseData = WORKFLOW_PHASES.find((p) => p.id === currentPhase);
  const completion = getPhaseCompletion(currentPhase);
  const iterationCount = getIterationCount();

  if (!currentPhaseData) return null;

  return (
    <div className="flex items-center gap-3 text-sm">
      <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-50 rounded-full">
        <currentPhaseData.icon className="h-4 w-4 text-blue-600" />
        <span className="text-blue-700 font-medium">{currentPhaseData.label}</span>
      </div>
      {iterationCount > 0 && (
        <span className="text-gray-500">Iteration {iterationCount}</span>
      )}
      {completion.items.length > 0 && (
        <span className="text-green-600">
          <CheckIcon className="h-4 w-4 inline mr-1" />
          {completion.items.length} complete
        </span>
      )}
    </div>
  );
}
