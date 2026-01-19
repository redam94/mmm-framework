import { Button } from '@tremor/react';
import { ArrowLeftIcon, ArrowRightIcon } from '@heroicons/react/24/outline';
import { useConfigWizardStore } from '../../../stores/configWizardStore';

interface WizardNavigationProps {
  onClose?: () => void;
}

export function WizardNavigation({ onClose }: WizardNavigationProps) {
  const {
    currentStep,
    nextStep,
    prevStep,
    isFirstStep,
    isLastStep,
    canProceed,
    stepValidation,
  } = useConfigWizardStore();

  const currentValidation = stepValidation[currentStep];
  const canGoNext = canProceed();

  return (
    <div className="flex items-center justify-between pt-6 border-t mt-6">
      <div className="flex items-center gap-3">
        {onClose && (
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
        )}
        {!isFirstStep() && (
          <Button variant="secondary" icon={ArrowLeftIcon} onClick={prevStep}>
            Previous
          </Button>
        )}
      </div>

      <div className="flex items-center gap-3">
        {/* Show validation status */}
        {!canGoNext && currentValidation.errors.length > 0 && (
          <span className="text-sm text-red-600">
            {currentValidation.errors[0]}
          </span>
        )}

        {!isLastStep() && (
          <Button
            icon={ArrowRightIcon}
            iconPosition="right"
            onClick={nextStep}
            disabled={!canGoNext && currentStep !== 'controls'} // Controls step is optional
          >
            Next
          </Button>
        )}
      </div>
    </div>
  );
}

export default WizardNavigation;
