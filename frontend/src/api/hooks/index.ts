// Budget plan hooks (Planner persistence)
export {
  budgetPlanKeys,
  useBudgetPlans,
  useBudgetPlan,
  useSaveBudgetPlan,
  useDeleteBudgetPlan,
  budgetPlanCsvUrl,
} from './useBudgetPlans';
export type { BudgetPlanInfo, BudgetPlanUpsertRequest } from './useBudgetPlans';

// Planner compute hooks (non-blocking optimize / what-if jobs)
export { plannerKeys, usePlannerOptimization, usePlannerScenario } from './usePlanner';

// Project hooks
export {
  projectKeys,
  useProjects,
  useProject,
  useCreateProject,
  useUpdateProject,
  useOnboardProject,
  useDeleteProject,
} from './useProjects';

// Knowledge base hooks
export {
  kbKeys,
  useKbDocuments,
  useUploadKbDocument,
  useDeleteKbDocument,
  useKbSearch,
} from './useKb';

// Team hooks
export {
  teamKeys,
  useUsers,
  useCreateUser,
  useUpdateUser,
  useDeleteUser,
  useProjectMembers,
  useSetProjectMembers,
} from './useTeam';

// Session hooks
export {
  sessionKeys,
  useSessions,
  useSession,
  useCreateSession,
  useUpdateSession,
  useDeleteSession,
  analysisPlansKeys,
  useAnalysisPlans,
} from './useSessions';
export type { AnalysisPlanInfo } from './useSessions';

// Model Garden hooks
export {
  gardenKeys,
  useGardenModels,
  useGardenVersions,
  useGardenModel,
  useGardenSource,
  useRegisterGardenModel,
  usePromoteGardenModel,
  useUpdateGardenDocs,
  useDeleteGardenModel,
  useGardenTest,
} from './useModelGarden';

// Atelier notebook hooks
export { notebookKeys, useNotebookDoc, useSaveNotebook } from './useAtelierNotebook';

// Atelier copilot chat hooks
export {
  copilotChatKeys,
  useCopilotChat,
  useSaveCopilotChat,
  useCopilotChatState,
  type PersistedMsg,
} from './useCopilotChat';
