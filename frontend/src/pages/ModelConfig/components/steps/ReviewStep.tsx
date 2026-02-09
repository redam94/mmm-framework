import { useEffect, useState } from 'react';
import { Card, Title, Text, Badge, Button } from '@tremor/react';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';
import { useConfigWizardStore, STEP_LABELS } from '../../../../stores/configWizardStore';
import { useCreateConfig, useValidateConfig } from '../../../../api/hooks';
import { LoadingSpinner } from '../../../../components/common/LoadingSpinner';
import { transformToConfigRequest } from '../../utils/transformers';

export function ReviewStep() {
  const {
    draftConfig,
    stepValidation,
    setStepValidation,
    closeWizard,
    resetWizard,
    editingConfigId,
  } = useConfigWizardStore();

  const createConfig = useCreateConfig();
  const validateConfig = useValidateConfig();
  const [validationRan, setValidationRan] = useState(false);

  // Run validation on mount
  useEffect(() => {
    if (!validationRan && draftConfig.name && draftConfig.kpi?.name && draftConfig.media_channels?.length) {
      const request = transformToConfigRequest(draftConfig);
      validateConfig.mutate(request);
      setValidationRan(true);
    }
  }, [draftConfig, validateConfig, validationRan]);

  // Update step validation based on API validation
  useEffect(() => {
    if (validateConfig.isSuccess) {
      const result = validateConfig.data;
      setStepValidation('review', result.valid, result.errors || []);
    } else if (validateConfig.isError) {
      setStepValidation('review', false, ['Validation failed']);
    }
  }, [validateConfig.isSuccess, validateConfig.isError, validateConfig.data, setStepValidation]);

  const handleSave = async () => {
    const request = transformToConfigRequest(draftConfig);
    try {
      await createConfig.mutateAsync(request);
      resetWizard();
      closeWizard();
    } catch (error) {
      // Error is handled by mutation
    }
  };

  const kpi = draftConfig.kpi;
  const mediaChannels = draftConfig.media_channels || [];
  const controls = draftConfig.controls || [];
  const settings = draftConfig.model_settings;

  return (
    <div className="space-y-6">
      {/* Validation Status */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <Title className="text-sm">Configuration Validation</Title>
          <Button
            variant="secondary"
            size="xs"
            onClick={() => {
              setValidationRan(false);
              const request = transformToConfigRequest(draftConfig);
              validateConfig.mutate(request);
              setValidationRan(true);
            }}
            loading={validateConfig.isPending}
          >
            Re-validate
          </Button>
        </div>

        {validateConfig.isPending && (
          <div className="flex items-center gap-2 text-gray-600">
            <LoadingSpinner size="sm" />
            <Text>Validating configuration...</Text>
          </div>
        )}

        {validateConfig.isSuccess && (
          <div className="space-y-3">
            {validateConfig.data.valid ? (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircleIcon className="h-5 w-5" />
                <Text className="text-green-600 font-medium">Configuration is valid</Text>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-red-600">
                <XCircleIcon className="h-5 w-5" />
                <Text className="text-red-600 font-medium">Configuration has errors</Text>
              </div>
            )}

            {validateConfig.data.errors?.length > 0 && (
              <div className="mt-2 p-3 bg-red-50 rounded-lg">
                <Text className="font-medium text-red-800 mb-2">Errors:</Text>
                <ul className="list-disc list-inside text-sm text-red-700 space-y-1">
                  {validateConfig.data.errors.map((error, i) => (
                    <li key={i}>{error}</li>
                  ))}
                </ul>
              </div>
            )}

            {validateConfig.data.warnings?.length > 0 && (
              <div className="mt-2 p-3 bg-yellow-50 rounded-lg">
                <Text className="font-medium text-yellow-800 mb-2">Warnings:</Text>
                <ul className="list-disc list-inside text-sm text-yellow-700 space-y-1">
                  {validateConfig.data.warnings.map((warning, i) => (
                    <li key={i}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {validateConfig.isError && (
          <div className="flex items-center gap-2 text-red-600">
            <ExclamationTriangleIcon className="h-5 w-5" />
            <Text className="text-red-600">
              Validation failed: {(validateConfig.error as Error).message}
            </Text>
          </div>
        )}
      </Card>

      {/* Configuration Summary */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Configuration Summary</Title>

        <div className="space-y-4">
          {/* Basic Info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Text className="text-xs text-gray-500">Configuration Name</Text>
              <Text className="font-medium">{draftConfig.name || 'Unnamed'}</Text>
            </div>
            <div>
              <Text className="text-xs text-gray-500">Description</Text>
              <Text className="font-medium">{draftConfig.description || 'None'}</Text>
            </div>
          </div>

          <hr />

          {/* KPI */}
          <div>
            <Text className="text-xs text-gray-500 mb-2">KPI Configuration</Text>
            <div className="flex flex-wrap gap-2">
              <Badge color="blue">{kpi?.name || 'Not set'}</Badge>
              {kpi?.log_transform && <Badge color="gray">Log-transformed</Badge>}
              <Badge color="gray">Dimensions: {kpi?.dimensions?.join(', ') || 'Period'}</Badge>
            </div>
          </div>

          <hr />

          {/* Media Channels */}
          <div>
            <Text className="text-xs text-gray-500 mb-2">
              Media Channels ({mediaChannels.length})
            </Text>
            <div className="flex flex-wrap gap-2">
              {mediaChannels.map((ch, i) => (
                <Badge key={i} color="green">
                  {ch.name || `Channel ${i + 1}`}
                </Badge>
              ))}
              {mediaChannels.length === 0 && (
                <Badge color="red">No channels configured</Badge>
              )}
            </div>
          </div>

          <hr />

          {/* Controls */}
          <div>
            <Text className="text-xs text-gray-500 mb-2">
              Control Variables ({controls.length})
            </Text>
            <div className="flex flex-wrap gap-2">
              {controls.map((c, i) => (
                <Badge key={i} color="purple">
                  {c.name || `Control ${i + 1}`}
                </Badge>
              ))}
              {controls.length === 0 && (
                <Badge color="gray">None</Badge>
              )}
            </div>
          </div>

          <hr />

          {/* Model Settings */}
          <div>
            <Text className="text-xs text-gray-500 mb-2">Model Settings</Text>
            <div className="flex flex-wrap gap-2">
              <Badge color="gray">
                {settings?.inference_method === 'bayesian_numpyro' ? 'NumPyro' : 'PyMC'}
              </Badge>
              <Badge color="gray">
                {settings?.n_chains || 4} chains × {settings?.n_draws || 2000} draws
              </Badge>
              <Badge color="gray">Trend: {settings?.trend?.type || 'none'}</Badge>
              {settings?.hierarchical?.enabled && <Badge color="gray">Hierarchical</Badge>}
            </div>
          </div>
        </div>
      </Card>

      {/* Step Validation Summary */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Step Validation</Title>
        <div className="grid grid-cols-2 gap-3">
          {(Object.keys(stepValidation) as Array<keyof typeof stepValidation>).map((step) => {
            if (step === 'review') return null;
            const validation = stepValidation[step];
            return (
              <div key={step} className="flex items-center gap-2">
                {validation.valid ? (
                  <CheckCircleIcon className="h-5 w-5 text-green-500" />
                ) : (
                  <XCircleIcon className="h-5 w-5 text-red-500" />
                )}
                <Text className={validation.valid ? 'text-green-700' : 'text-red-700'}>
                  {STEP_LABELS[step]}
                </Text>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Save Button */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <Title className="text-sm">
              {editingConfigId ? 'Update Configuration' : 'Save Configuration'}
            </Title>
            <Text className="text-gray-600 text-sm">
              {validateConfig.data?.valid
                ? 'Your configuration is ready to be saved.'
                : 'Please fix validation errors before saving.'}
            </Text>
          </div>
          <Button
            onClick={handleSave}
            loading={createConfig.isPending}
            disabled={!validateConfig.data?.valid}
          >
            {editingConfigId ? 'Update' : 'Save'} Configuration
          </Button>
        </div>

        {createConfig.isError && (
          <div className="mt-4 p-3 bg-red-50 rounded-lg">
            <Text className="text-red-600">
              Failed to save: {(createConfig.error as Error).message}
            </Text>
          </div>
        )}
      </Card>
    </div>
  );
}

export default ReviewStep;
