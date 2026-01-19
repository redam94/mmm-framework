import { Card, Title, Text, Select, SelectItem, Badge, Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell } from '@tremor/react';
import { ExclamationTriangleIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import { useModels, useModelResults } from '../../api/hooks';
import { useProjectStore } from '../../stores/projectStore';
import { LoadingPage, LoadingSpinner } from '../../components/common/LoadingSpinner';

// Diagnostic thresholds
const THRESHOLDS = {
  rhat: { warning: 1.01, critical: 1.05 },
  ess_bulk: { warning: 400, critical: 100 },
  ess_tail: { warning: 400, critical: 100 },
  divergences: { warning: 1, critical: 10 },
};

// Diagnostic alert component
function DiagnosticAlert({
  title,
  message,
  severity,
}: {
  title: string;
  message: string;
  severity: 'warning' | 'critical' | 'success';
}) {
  const colorMap = {
    warning: 'bg-yellow-50 border-yellow-200',
    critical: 'bg-red-50 border-red-200',
    success: 'bg-green-50 border-green-200',
  };

  const iconColorMap = {
    warning: 'text-yellow-500',
    critical: 'text-red-500',
    success: 'text-green-500',
  };

  const Icon = severity === 'success' ? CheckCircleIcon : ExclamationTriangleIcon;

  return (
    <div className={`p-4 rounded-lg border ${colorMap[severity]}`}>
      <div className="flex items-start gap-3">
        <Icon className={`h-5 w-5 mt-0.5 ${iconColorMap[severity]}`} />
        <div>
          <Text className="font-medium">{title}</Text>
          <Text className="text-sm text-gray-600 mt-1">{message}</Text>
        </div>
      </div>
    </div>
  );
}

// Results display component
function ModelDiagnosticsDisplay({ modelId }: { modelId: string }) {
  const { data: results, isLoading } = useModelResults(modelId);

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (!results) {
    return <Text className="text-gray-500">No results available</Text>;
  }

  const { diagnostics, parameters } = results;
  const alerts: { title: string; message: string; severity: 'warning' | 'critical' | 'success' }[] = [];

  // Check diagnostics
  if (diagnostics.n_divergences === 0 && diagnostics.rhat_max < 1.01) {
    alerts.push({
      title: 'All Diagnostics Pass',
      message: 'No convergence issues detected. The model appears to have sampled well.',
      severity: 'success',
    });
  } else {
    if (diagnostics.n_divergences > THRESHOLDS.divergences.critical) {
      alerts.push({
        title: 'High Number of Divergent Transitions',
        message: `${diagnostics.n_divergences} divergences detected. This indicates the sampler had difficulty exploring the posterior.`,
        severity: 'critical',
      });
    } else if (diagnostics.n_divergences > THRESHOLDS.divergences.warning) {
      alerts.push({
        title: 'Some Divergent Transitions',
        message: `${diagnostics.n_divergences} divergence(s) detected. Consider reparameterizing or increasing target_accept.`,
        severity: 'warning',
      });
    }

    if (diagnostics.rhat_max > THRESHOLDS.rhat.critical) {
      alerts.push({
        title: 'Chains May Not Have Converged',
        message: `Maximum R-hat of ${diagnostics.rhat_max.toFixed(3)} detected. Run more iterations.`,
        severity: 'critical',
      });
    } else if (diagnostics.rhat_max > THRESHOLDS.rhat.warning) {
      alerts.push({
        title: 'R-hat Slightly Elevated',
        message: `Maximum R-hat of ${diagnostics.rhat_max.toFixed(3)}. Consider running more iterations.`,
        severity: 'warning',
      });
    }

    if (diagnostics.ess_bulk_min < THRESHOLDS.ess_bulk.critical) {
      alerts.push({
        title: 'Very Low Effective Sample Size',
        message: `Minimum ESS bulk of ${diagnostics.ess_bulk_min.toFixed(0)}. Results may be unreliable.`,
        severity: 'critical',
      });
    } else if (diagnostics.ess_bulk_min < THRESHOLDS.ess_bulk.warning) {
      alerts.push({
        title: 'Low Effective Sample Size',
        message: `Minimum ESS bulk of ${diagnostics.ess_bulk_min.toFixed(0)}. Consider more samples.`,
        severity: 'warning',
      });
    }
  }

  return (
    <div className="space-y-6">
      {/* Alerts */}
      <div className="space-y-3">
        {alerts.map((alert, i) => (
          <DiagnosticAlert key={i} {...alert} />
        ))}
      </div>

      {/* Summary stats */}
      <Card>
        <Title className="text-sm">Diagnostics Summary</Title>
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <Text className="text-xs text-gray-500">Divergences</Text>
            <Text className="text-2xl font-bold">{diagnostics.n_divergences}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500">Max R-hat</Text>
            <Text className="text-2xl font-bold">{diagnostics.rhat_max.toFixed(3)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500">Min ESS (bulk)</Text>
            <Text className="text-2xl font-bold">{diagnostics.ess_bulk_min.toFixed(0)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500">Min ESS (tail)</Text>
            <Text className="text-2xl font-bold">{diagnostics.ess_tail_min.toFixed(0)}</Text>
          </div>
        </div>
      </Card>

      {/* Parameter table */}
      <Card>
        <Title className="text-sm">Parameter Convergence</Title>
        <div className="mt-4 overflow-x-auto">
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell>Parameter</TableHeaderCell>
                <TableHeaderCell>Mean</TableHeaderCell>
                <TableHeaderCell>Std</TableHeaderCell>
                <TableHeaderCell>HDI Low</TableHeaderCell>
                <TableHeaderCell>HDI High</TableHeaderCell>
                <TableHeaderCell>R-hat</TableHeaderCell>
                <TableHeaderCell>ESS Bulk</TableHeaderCell>
                <TableHeaderCell>Status</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {parameters.slice(0, 20).map((p) => {
                const hasIssue = p.rhat > 1.01 || p.ess_bulk < 400;
                return (
                  <TableRow key={p.name}>
                    <TableCell className="font-mono text-sm">{p.name}</TableCell>
                    <TableCell>{p.mean.toFixed(3)}</TableCell>
                    <TableCell>{p.std.toFixed(3)}</TableCell>
                    <TableCell>{p.hdi_low.toFixed(3)}</TableCell>
                    <TableCell>{p.hdi_high.toFixed(3)}</TableCell>
                    <TableCell>
                      <span className={p.rhat > 1.01 ? 'text-red-600 font-medium' : ''}>
                        {p.rhat.toFixed(3)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className={p.ess_bulk < 400 ? 'text-yellow-600 font-medium' : ''}>
                        {p.ess_bulk.toFixed(0)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <Badge color={hasIssue ? 'yellow' : 'green'} size="xs">
                        {hasIssue ? 'Check' : 'OK'}
                      </Badge>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
        {parameters.length > 20 && (
          <Text className="mt-2 text-xs text-gray-500">
            Showing 20 of {parameters.length} parameters
          </Text>
        )}
      </Card>
    </div>
  );
}

export function DiagnosticsPage() {
  const { data: modelsData, isLoading } = useModels({ status: 'completed' });
  const { selectedModelId, setSelectedModel } = useProjectStore();

  if (isLoading) {
    return <LoadingPage message="Loading models..." />;
  }

  const models = modelsData?.models.filter((m) => m.status === 'completed') || [];

  return (
    <div className="space-y-6">
      <div>
        <Title>Model Diagnostics</Title>
        <Text>Review MCMC convergence and sampling diagnostics</Text>
      </div>

      {/* Model selector */}
      <Card>
        <Title className="text-sm">Select Model</Title>
        <div className="mt-4">
          <Select
            value={selectedModelId || ''}
            onValueChange={(v) => setSelectedModel(v)}
            placeholder="Select a completed model..."
          >
            {models.map((m) => (
              <SelectItem key={m.model_id} value={m.model_id}>
                {m.name || m.model_id}
              </SelectItem>
            ))}
          </Select>
        </div>
      </Card>

      {/* Diagnostics display */}
      {selectedModelId ? (
        <ModelDiagnosticsDisplay modelId={selectedModelId} />
      ) : (
        <Card>
          <Text className="text-gray-500 text-center">
            Select a completed model to view diagnostics
          </Text>
        </Card>
      )}
    </div>
  );
}

export default DiagnosticsPage;
