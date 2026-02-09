import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  KPIConfig,
  MediaChannelConfig,
  ControlVariableConfig,
  ModelSettings,
  MFFColumnConfig,
  DimensionAlignmentConfig,
  DataFrequency
} from '../api/types';

// Wizard steps
export type WizardStep =
  | 'kpi'
  | 'media_channels'
  | 'controls'
  | 'model_settings'
  | 'mff_columns'
  | 'alignment'
  | 'review';

export const WIZARD_STEPS: WizardStep[] = [
  'kpi',
  'media_channels',
  'controls',
  'model_settings',
  'mff_columns',
  'alignment',
  'review'
];

export const STEP_LABELS: Record<WizardStep, string> = {
  kpi: 'KPI',
  media_channels: 'Media Channels',
  controls: 'Controls',
  model_settings: 'Model Settings',
  mff_columns: 'Column Mapping',
  alignment: 'Alignment',
  review: 'Review',
};

// Draft configuration type (matches ConfigCreateRequest structure)
export interface WizardDraftConfig {
  name: string;
  description?: string;

  // MFF Config parts
  kpi: KPIConfig;
  media_channels: MediaChannelConfig[];
  controls: ControlVariableConfig[];
  columns: MFFColumnConfig;
  alignment: DimensionAlignmentConfig;
  date_format: string;
  frequency: DataFrequency;
  fill_missing_media: number;
  fill_missing_controls?: number;

  // Model settings
  model_settings: ModelSettings;
}

// Step validation state
interface StepValidation {
  valid: boolean;
  errors: string[];
}

interface ConfigWizardState {
  // Wizard UI state
  currentStep: WizardStep;
  isOpen: boolean;
  editingConfigId: string | null;

  // Draft data (persisted between steps)
  draftConfig: Partial<WizardDraftConfig>;

  // Validation state per step
  stepValidation: Record<WizardStep, StepValidation>;

  // Actions
  openWizard: (editingConfigId?: string | null, initialData?: Partial<WizardDraftConfig>) => void;
  closeWizard: () => void;
  setStep: (step: WizardStep) => void;
  nextStep: () => void;
  prevStep: () => void;
  updateDraft: (data: Partial<WizardDraftConfig>) => void;
  setStepValidation: (step: WizardStep, valid: boolean, errors?: string[]) => void;
  resetWizard: () => void;
  canProceed: () => boolean;
  getStepIndex: () => number;
  isFirstStep: () => boolean;
  isLastStep: () => boolean;
}

// Default draft configuration
export const DEFAULT_DRAFT_CONFIG: Partial<WizardDraftConfig> = {
  name: '',
  description: '',
  kpi: {
    name: '',
    dimensions: ['Period'],
    log_transform: false,
    floor_value: 0,
  },
  media_channels: [],
  controls: [],
  columns: {
    period: 'Period',
    geography: 'Geography',
    product: 'Product',
    campaign: 'Campaign',
    outlet: 'Outlet',
    creative: 'Creative',
    variable_name: 'VariableName',
    variable_value: 'VariableValue',
  },
  alignment: {
    geo_allocation: 'equal',
    product_allocation: 'equal',
    prefer_disaggregation: true,
  },
  date_format: '%Y-%m-%d',
  frequency: 'W',
  fill_missing_media: 0,
  model_settings: {
    inference_method: 'bayesian_numpyro',
    n_chains: 4,
    n_draws: 2000,
    n_tune: 1000,
    target_accept: 0.9,
    trend: { type: 'none' },
    seasonality: { yearly_order: 2, monthly_order: 0, weekly_order: 0 },
    hierarchical: {
      enabled: false,
      pool_across_geo: true,
      pool_across_product: true,
      non_centered: true,
    },
  },
};

const DEFAULT_STEP_VALIDATION: Record<WizardStep, StepValidation> = {
  kpi: { valid: false, errors: [] },
  media_channels: { valid: false, errors: [] },
  controls: { valid: true, errors: [] }, // Optional step
  model_settings: { valid: true, errors: [] }, // Has defaults
  mff_columns: { valid: true, errors: [] }, // Has defaults
  alignment: { valid: true, errors: [] }, // Has defaults
  review: { valid: false, errors: [] },
};

export const useConfigWizardStore = create<ConfigWizardState>()(
  persist(
    (set, get) => ({
      currentStep: 'kpi',
      isOpen: false,
      editingConfigId: null,
      draftConfig: { ...DEFAULT_DRAFT_CONFIG },
      stepValidation: { ...DEFAULT_STEP_VALIDATION },

      openWizard: (editingConfigId = null, initialData) => {
        set({
          isOpen: true,
          editingConfigId,
          currentStep: 'kpi',
          draftConfig: initialData || { ...DEFAULT_DRAFT_CONFIG },
          stepValidation: { ...DEFAULT_STEP_VALIDATION },
        });
      },

      closeWizard: () => {
        set({
          isOpen: false,
          editingConfigId: null,
        });
      },

      setStep: (step) => {
        set({ currentStep: step });
      },

      nextStep: () => {
        const { currentStep, stepValidation } = get();
        const currentIndex = WIZARD_STEPS.indexOf(currentStep);

        // Check if current step is valid before proceeding
        if (!stepValidation[currentStep].valid && currentStep !== 'controls') {
          return;
        }

        if (currentIndex < WIZARD_STEPS.length - 1) {
          set({ currentStep: WIZARD_STEPS[currentIndex + 1] });
        }
      },

      prevStep: () => {
        const { currentStep } = get();
        const currentIndex = WIZARD_STEPS.indexOf(currentStep);
        if (currentIndex > 0) {
          set({ currentStep: WIZARD_STEPS[currentIndex - 1] });
        }
      },

      updateDraft: (data) => {
        set((state) => ({
          draftConfig: { ...state.draftConfig, ...data },
        }));
      },

      setStepValidation: (step, valid, errors = []) => {
        set((state) => ({
          stepValidation: {
            ...state.stepValidation,
            [step]: { valid, errors },
          },
        }));
      },

      resetWizard: () => {
        set({
          currentStep: 'kpi',
          isOpen: false,
          editingConfigId: null,
          draftConfig: { ...DEFAULT_DRAFT_CONFIG },
          stepValidation: { ...DEFAULT_STEP_VALIDATION },
        });
      },

      canProceed: () => {
        const { currentStep, stepValidation } = get();
        // Controls step is optional
        if (currentStep === 'controls') return true;
        return stepValidation[currentStep].valid;
      },

      getStepIndex: () => {
        const { currentStep } = get();
        return WIZARD_STEPS.indexOf(currentStep);
      },

      isFirstStep: () => {
        const { currentStep } = get();
        return currentStep === WIZARD_STEPS[0];
      },

      isLastStep: () => {
        const { currentStep } = get();
        return currentStep === WIZARD_STEPS[WIZARD_STEPS.length - 1];
      },
    }),
    {
      name: 'mmm-config-wizard',
      partialize: (state) => ({
        draftConfig: state.draftConfig,
        currentStep: state.currentStep,
        isOpen: state.isOpen,
        editingConfigId: state.editingConfigId,
      }),
    }
  )
);
