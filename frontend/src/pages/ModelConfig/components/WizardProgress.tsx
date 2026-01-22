import { CheckIcon } from '@heroicons/react/24/solid';
import { useConfigWizardStore, WIZARD_STEPS, STEP_LABELS } from '../../../stores/configWizardStore';

export function WizardProgress() {
  const { currentStep, stepValidation, setStep } = useConfigWizardStore();
  const currentIndex = WIZARD_STEPS.indexOf(currentStep);

  return (
    <nav aria-label="Progress" className="mb-8">
      <ol className="flex items-center">
        {WIZARD_STEPS.map((step, index) => {
          const isComplete = stepValidation[step].valid;
          const isCurrent = step === currentStep;
          const isPast = index < currentIndex;

          return (
            <li
              key={step}
              className={`relative ${index !== WIZARD_STEPS.length - 1 ? 'pr-8 sm:pr-20 flex-1' : ''}`}
            >
              {/* Connector line */}
              {index !== WIZARD_STEPS.length - 1 && (
                <div
                  className="absolute top-4 left-8 -ml-px mt-0.5 h-0.5 w-full bg-gray-200"
                  aria-hidden="true"
                >
                  <div
                    className={`h-full transition-all duration-300 ${
                      isPast || isComplete ? 'bg-blue-600' : 'bg-gray-200'
                    }`}
                    style={{ width: isPast ? '100%' : '0%' }}
                  />
                </div>
              )}

              {/* Step circle and label */}
              <button
                type="button"
                onClick={() => {
                  // Allow navigation to past steps or current step
                  if (index <= currentIndex) {
                    setStep(step);
                  }
                }}
                className={`group relative flex items-center ${
                  index <= currentIndex ? 'cursor-pointer' : 'cursor-not-allowed'
                }`}
                disabled={index > currentIndex}
              >
                <span className="flex h-9 items-center" aria-hidden="true">
                  <span
                    className={`relative z-10 flex h-8 w-8 items-center justify-center rounded-full transition-colors ${
                      isComplete
                        ? 'bg-blue-600 group-hover:bg-blue-700'
                        : isCurrent
                        ? 'border-2 border-blue-600 bg-white'
                        : 'border-2 border-gray-300 bg-white group-hover:border-gray-400'
                    }`}
                  >
                    {isComplete ? (
                      <CheckIcon className="h-5 w-5 text-white" />
                    ) : (
                      <span
                        className={`h-2.5 w-2.5 rounded-full ${
                          isCurrent ? 'bg-blue-600' : 'bg-transparent'
                        }`}
                      />
                    )}
                  </span>
                </span>
                <span className="ml-3 hidden sm:block">
                  <span
                    className={`text-sm font-medium ${
                      isCurrent ? 'text-blue-600' : isComplete ? 'text-gray-900' : 'text-gray-500'
                    }`}
                  >
                    {STEP_LABELS[step]}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export default WizardProgress;
