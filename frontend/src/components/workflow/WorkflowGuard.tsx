import { Navigate } from 'react-router-dom';
import { Card, Title, Text, Button, Badge } from '@tremor/react';
import { ExclamationTriangleIcon, ArrowRightIcon } from '@heroicons/react/24/outline';
import { useWorkflowStore, type BayesianPhase } from '../../stores/workflowStore';

interface WorkflowGuardProps {
  requiredPhase: BayesianPhase;
  children: React.ReactNode;
  redirectTo?: string;
  showWarning?: boolean;
}

// Phase order for comparison
const PHASE_ORDER: BayesianPhase[] = [
  'planning',
  'data',
  'config',
  'prior_predictive',
  'fit',
  'diagnostics',
  'posterior_predictive',
  'model_comparison',
  'sensitivity',
  'results',
];

const PHASE_PATHS: Record<BayesianPhase, string> = {
  planning: '/planning',
  data: '/data',
  config: '/config',
  prior_predictive: '/config',
  fit: '/fit',
  diagnostics: '/diagnostics',
  posterior_predictive: '/diagnostics',
  model_comparison: '/results',
  sensitivity: '/results',
  results: '/results',
};

export function WorkflowGuard({
  requiredPhase,
  children,
  redirectTo,
  showWarning = true,
}: WorkflowGuardProps) {
  const { currentPhase, getPhaseCompletion } = useWorkflowStore();

  const currentIndex = PHASE_ORDER.indexOf(currentPhase);
  const requiredIndex = PHASE_ORDER.indexOf(requiredPhase);

  // Check if required phase is ahead of current phase
  const isBlocked = requiredIndex > currentIndex + 1;

  // Get completion status of prerequisite phase
  const prereqPhase = requiredIndex > 0 ? PHASE_ORDER[requiredIndex - 1] : null;
  const prereqCompletion = prereqPhase ? getPhaseCompletion(prereqPhase) : null;
  const prereqComplete = prereqCompletion?.complete ?? true;

  // If blocked and redirect specified, redirect
  if (isBlocked && redirectTo) {
    return <Navigate to={redirectTo} replace />;
  }

  // If blocked and showing warning
  if (isBlocked && showWarning) {
    return (
      <Card className="max-w-lg mx-auto mt-8">
        <div className="flex items-center gap-3 text-yellow-600 mb-4">
          <ExclamationTriangleIcon className="h-8 w-8" />
          <Title>Workflow Step Required</Title>
        </div>
        <Text className="mb-4">
          You need to complete the previous workflow steps before accessing this page.
          The Bayesian workflow requires completing each phase in order to ensure
          scientific rigor.
        </Text>
        <div className="flex items-center gap-2 mb-4">
          <Badge color="blue">{currentPhase}</Badge>
          <ArrowRightIcon className="h-4 w-4 text-gray-400" />
          <Badge color="gray">{requiredPhase}</Badge>
        </div>
        <Button onClick={() => window.location.href = PHASE_PATHS[currentPhase]}>
          Go to {currentPhase.replace('_', ' ')} phase
        </Button>
      </Card>
    );
  }

  // Show warning if prerequisite not complete but allow access
  if (!prereqComplete && showWarning) {
    return (
      <div>
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-center gap-3">
          <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600" />
          <Text className="text-yellow-800">
            The previous phase ({prereqPhase?.replace('_', ' ')}) is not yet complete.
            Consider completing it before proceeding.
          </Text>
        </div>
        {children}
      </div>
    );
  }

  return <>{children}</>;
}

// Hook to check workflow progression
export function useWorkflowCheck(requiredPhase: BayesianPhase) {
  const { currentPhase, getPhaseCompletion } = useWorkflowStore();

  const currentIndex = PHASE_ORDER.indexOf(currentPhase);
  const requiredIndex = PHASE_ORDER.indexOf(requiredPhase);

  const isBlocked = requiredIndex > currentIndex + 1;
  const prereqPhase = requiredIndex > 0 ? PHASE_ORDER[requiredIndex - 1] : null;
  const prereqCompletion = prereqPhase ? getPhaseCompletion(prereqPhase) : null;

  return {
    isBlocked,
    currentPhase,
    requiredPhase,
    prereqPhase,
    prereqComplete: prereqCompletion?.complete ?? true,
  };
}

// Pre-specification warning component
interface PreSpecificationWarningProps {
  show: boolean;
  onAcknowledge: () => void;
}

export function PreSpecificationWarning({ show, onAcknowledge }: PreSpecificationWarningProps) {
  if (!show) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <Card className="max-w-md">
        <div className="flex items-center gap-3 text-red-600 mb-4">
          <ExclamationTriangleIcon className="h-8 w-8" />
          <Title>Pre-Specification Warning</Title>
        </div>
        <Text className="mb-4">
          You are modifying the model configuration after seeing results. This may
          introduce bias through specification searching (p-hacking). Changes should
          be clearly documented and acknowledged.
        </Text>
        <Text className="text-sm text-gray-500 mb-4">
          To maintain scientific rigor, please document why you are making this change
          and acknowledge that this is a post-hoc modification.
        </Text>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => window.history.back()}>
            Go Back
          </Button>
          <Button color="red" onClick={onAcknowledge}>
            I Understand - Proceed
          </Button>
        </div>
      </Card>
    </div>
  );
}

// Iteration counter display
export function IterationCounter() {
  const { getIterationCount } = useWorkflowStore();
  const count = getIterationCount();

  if (count === 0) return null;

  return (
    <div className="flex items-center gap-2 text-sm">
      <Badge color={count > 3 ? 'yellow' : 'gray'}>
        Iteration {count}
      </Badge>
      {count > 3 && (
        <Text className="text-xs text-yellow-600">
          Many iterations may indicate specification searching
        </Text>
      )}
    </div>
  );
}
