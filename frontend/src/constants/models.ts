export type Provider = 'anthropic' | 'openai' | 'google';

export interface ModelOption {
  id: string;
  label: string;
  provider: Provider;
  description: string;
}

export const MODELS: ModelOption[] = [
  {
    id: 'claude-sonnet-4-6',
    label: 'Claude Sonnet 4.6',
    provider: 'anthropic',
    description: 'Fast, capable — recommended',
  },
  {
    id: 'claude-opus-4-5',
    label: 'Claude Opus 4.5',
    provider: 'anthropic',
    description: 'Most capable Anthropic model',
  },
  {
    id: 'gpt-4o',
    label: 'GPT-4o',
    provider: 'openai',
    description: 'OpenAI flagship model',
  },
  {
    id: 'gpt-4o-mini',
    label: 'GPT-4o mini',
    provider: 'openai',
    description: 'Fast and cost-efficient',
  },
  {
    id: 'gemini-2.0-flash',
    label: 'Gemini 2.0 Flash',
    provider: 'google',
    description: 'Fast Google model',
  },
  {
    id: 'gemini-2.5-pro',
    label: 'Gemini 2.5 Pro',
    provider: 'google',
    description: 'Most capable Google model',
  },
];

export const PROVIDER_LABELS: Record<Provider, string> = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  google: 'Google',
};

export function getProviderForModel(modelId: string): Provider {
  const lower = modelId.toLowerCase();
  if (lower.includes('claude')) return 'anthropic';
  if (lower.includes('gpt') || lower.includes('o1') || lower.includes('o3')) return 'openai';
  return 'google';
}

export function getModelLabel(modelId: string): string {
  return MODELS.find((m) => m.id === modelId)?.label ?? modelId;
}

export function getModelsByProvider(): Record<Provider, ModelOption[]> {
  return MODELS.reduce(
    (acc, m) => {
      acc[m.provider].push(m);
      return acc;
    },
    { anthropic: [], openai: [], google: [] } as Record<Provider, ModelOption[]>
  );
}
