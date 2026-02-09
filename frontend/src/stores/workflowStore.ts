import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Bayesian workflow phases
export type BayesianPhase =
  | 'planning'
  | 'data'
  | 'config'
  | 'prior_predictive'
  | 'fit'
  | 'diagnostics'
  | 'posterior_predictive'
  | 'model_comparison'
  | 'sensitivity'
  | 'results';

// Model iteration tracking
export interface ModelIteration {
  id: string;
  modelId: string;
  configId: string;
  dataId: string;
  phase: BayesianPhase;
  createdAt: string;
  status: 'running' | 'completed' | 'failed';
  diagnosticsPassed: boolean | null;
  notes: string;
}

// Workflow decision tracking (for scientific rigor)
export interface WorkflowDecision {
  id: string;
  phase: BayesianPhase;
  type: string; // e.g., 'config_change', 'model_expansion', 'prior_adjustment'
  rationale: string;
  timestamp: string;
  iteration: number;
  modelId?: string;
}

// Phase completion status
export interface PhaseCompletion {
  phase: BayesianPhase;
  complete: boolean;
  items: string[];
  warnings: string[];
}

interface WorkflowState {
  currentPhase: BayesianPhase;
  iterations: ModelIteration[];
  decisions: WorkflowDecision[];
  currentProjectId: string | null;

  // Actions
  setPhase: (phase: BayesianPhase) => void;
  addIteration: (iteration: Omit<ModelIteration, 'id' | 'createdAt'>) => string;
  updateIteration: (id: string, updates: Partial<ModelIteration>) => void;
  logDecision: (decision: Omit<WorkflowDecision, 'id' | 'timestamp' | 'iteration'>) => void;
  setProject: (projectId: string | null) => void;
  getPhaseCompletion: (phase: BayesianPhase) => PhaseCompletion;
  getIterationCount: () => number;
  clearWorkflow: () => void;

  // Validation
  canProceedToPhase: (phase: BayesianPhase) => { allowed: boolean; reason?: string };
  hasSpecificationShopping: () => boolean;
}

const generateId = () => Math.random().toString(36).substring(2, 15);

// Phase order for validation
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

export const useWorkflowStore = create<WorkflowState>()(
  persist(
    (set, get) => ({
      currentPhase: 'planning',
      iterations: [],
      decisions: [],
      currentProjectId: null,

      setPhase: (phase) => set({ currentPhase: phase }),

      addIteration: (iteration) => {
        const id = generateId();
        const newIteration: ModelIteration = {
          ...iteration,
          id,
          createdAt: new Date().toISOString(),
        };
        set((state) => ({
          iterations: [...state.iterations, newIteration],
        }));
        return id;
      },

      updateIteration: (id, updates) => {
        set((state) => ({
          iterations: state.iterations.map((it) =>
            it.id === id ? { ...it, ...updates } : it
          ),
        }));
      },

      logDecision: (decision) => {
        const state = get();

        // Warn if rationale is too short
        if (!decision.rationale || decision.rationale.length < 20) {
          console.warn(
            'Decision rationale should be at least 20 characters for scientific rigor'
          );
        }

        const newDecision: WorkflowDecision = {
          ...decision,
          id: generateId(),
          timestamp: new Date().toISOString(),
          iteration: state.iterations.length,
        };

        set((state) => ({
          decisions: [...state.decisions, newDecision],
        }));
      },

      setProject: (projectId) => set({ currentProjectId: projectId }),

      getPhaseCompletion: (phase) => {
        const state = get();
        const completion: PhaseCompletion = {
          phase,
          complete: false,
          items: [],
          warnings: [],
        };

        switch (phase) {
          case 'planning':
            // Planning is complete if we have logged a planning decision
            completion.complete = state.decisions.some((d) => d.phase === 'planning');
            if (completion.complete) {
              completion.items.push('Generative story defined');
            }
            break;

          case 'data':
            // Data is complete if we have at least one iteration with data
            completion.complete = state.iterations.length > 0;
            if (completion.complete) {
              completion.items.push('Dataset uploaded and validated');
            }
            break;

          case 'config':
            // Config is complete if we have a config decision
            completion.complete = state.decisions.some((d) => d.phase === 'config');
            if (completion.complete) {
              completion.items.push('Model configuration created');
            }
            break;

          case 'fit':
            // Fit is complete if we have a completed iteration
            const completedIterations = state.iterations.filter(
              (it) => it.status === 'completed'
            );
            completion.complete = completedIterations.length > 0;
            if (completion.complete) {
              completion.items.push(`${completedIterations.length} model(s) fitted`);
            }
            break;

          case 'diagnostics':
            // Diagnostics is complete if we've reviewed at least one model
            const reviewedModels = state.decisions.filter(
              (d) => d.phase === 'diagnostics'
            );
            completion.complete = reviewedModels.length > 0;
            if (completion.complete) {
              completion.items.push('Diagnostics reviewed');
            }
            // Warn if iterations failed diagnostics
            const failedDiag = state.iterations.filter(
              (it) => it.diagnosticsPassed === false
            );
            if (failedDiag.length > 0) {
              completion.warnings.push(
                `${failedDiag.length} model(s) had diagnostic issues`
              );
            }
            break;

          case 'results':
            completion.complete = state.decisions.some((d) => d.phase === 'results');
            break;

          default:
            break;
        }

        return completion;
      },

      getIterationCount: () => get().iterations.length,

      clearWorkflow: () =>
        set({
          currentPhase: 'planning',
          iterations: [],
          decisions: [],
          currentProjectId: null,
        }),

      canProceedToPhase: (phase) => {
        const state = get();
        const currentIndex = PHASE_ORDER.indexOf(state.currentPhase);
        const targetIndex = PHASE_ORDER.indexOf(phase);

        // Can always go back
        if (targetIndex <= currentIndex) {
          return { allowed: true };
        }

        // Check prerequisites for forward movement
        if (targetIndex > currentIndex + 1) {
          return {
            allowed: false,
            reason: `Complete ${PHASE_ORDER[currentIndex + 1]} phase first`,
          };
        }

        // Check phase-specific requirements
        const completion = state.getPhaseCompletion(state.currentPhase);
        if (!completion.complete && targetIndex > currentIndex) {
          return {
            allowed: false,
            reason: `Complete ${state.currentPhase} phase before proceeding`,
          };
        }

        return { allowed: true };
      },

      hasSpecificationShopping: () => {
        const state = get();

        // Check for pattern: many config changes after seeing results
        const configChangesAfterFit = state.decisions.filter(
          (d) => d.phase === 'config' && d.iteration > 0
        );

        // Warning if more than 5 config changes after initial fit
        if (configChangesAfterFit.length > 5) {
          return true;
        }

        // Check for short rationales (sign of specification shopping)
        const shortRationales = state.decisions.filter(
          (d) => d.rationale && d.rationale.length < 30
        );
        if (shortRationales.length > 3) {
          return true;
        }

        return false;
      },
    }),
    {
      name: 'mmm-workflow',
      partialize: (state) => ({
        currentPhase: state.currentPhase,
        iterations: state.iterations,
        decisions: state.decisions,
        currentProjectId: state.currentProjectId,
      }),
    }
  )
);
