// DAG starter templates for the CausalPlanner editor.
// Ported from stores/dagStore.ts TEMPLATES, normalized to the backend
// react_flow shape (`data: { label, variableName, type }`) so a template can
// be loaded straight into the editor and round-trip through PUT /dag/{thread}.

export interface TemplateNode {
  id: string;
  position: { x: number; y: number };
  data: { label: string; variableName: string; type: string };
}

export interface TemplateEdge {
  id: string;
  source: string;
  target: string;
}

export interface DagTemplate {
  id: 'simple' | 'mediation' | 'multivariate' | 'combined';
  name: string;
  description: string;
  nodes: TemplateNode[];
  edges: TemplateEdge[];
}

export const DAG_TEMPLATES: DagTemplate[] = [
  {
    id: 'simple',
    name: 'Simple MMM',
    description: 'Two media channels and a price control driving a single KPI.',
    nodes: [
      { id: 'kpi_1', position: { x: 400, y: 300 }, data: { label: 'Sales (KPI)', variableName: 'sales', type: 'kpi' } },
      { id: 'media_1', position: { x: 100, y: 100 }, data: { label: 'TV Spend', variableName: 'tv_spend', type: 'media' } },
      { id: 'media_2', position: { x: 250, y: 100 }, data: { label: 'Digital Spend', variableName: 'digital_spend', type: 'media' } },
      { id: 'control_1', position: { x: 550, y: 100 }, data: { label: 'Price', variableName: 'price', type: 'control' } },
    ],
    edges: [
      { id: 'e1', source: 'media_1', target: 'kpi_1' },
      { id: 'e2', source: 'media_2', target: 'kpi_1' },
      { id: 'e3', source: 'control_1', target: 'kpi_1' },
    ],
  },
  {
    id: 'mediation',
    name: 'With Mediation',
    description: 'Media drives awareness (mediator) which drives the KPI, plus a direct effect.',
    nodes: [
      { id: 'kpi_1', position: { x: 500, y: 400 }, data: { label: 'Sales (KPI)', variableName: 'sales', type: 'kpi' } },
      { id: 'media_1', position: { x: 100, y: 100 }, data: { label: 'TV Spend', variableName: 'tv_spend', type: 'media' } },
      { id: 'mediator_1', position: { x: 300, y: 250 }, data: { label: 'Brand Awareness', variableName: 'awareness', type: 'mediator' } },
    ],
    edges: [
      { id: 'e1', source: 'media_1', target: 'mediator_1' },
      { id: 'e2', source: 'mediator_1', target: 'kpi_1' },
      { id: 'e3', source: 'media_1', target: 'kpi_1' }, // Direct effect
    ],
  },
  {
    id: 'multivariate',
    name: 'Multi-Outcome',
    description: 'One media channel driving two outcomes with a cross-effect between them.',
    nodes: [
      { id: 'outcome_1', position: { x: 400, y: 300 }, data: { label: 'Revenue', variableName: 'revenue', type: 'outcome' } },
      { id: 'outcome_2', position: { x: 600, y: 300 }, data: { label: 'Volume', variableName: 'volume', type: 'outcome' } },
      { id: 'media_1', position: { x: 200, y: 100 }, data: { label: 'Marketing Spend', variableName: 'marketing', type: 'media' } },
    ],
    edges: [
      { id: 'e1', source: 'media_1', target: 'outcome_1' },
      { id: 'e2', source: 'media_1', target: 'outcome_2' },
      { id: 'e3', source: 'outcome_1', target: 'outcome_2' }, // Cross-effect
    ],
  },
  {
    id: 'combined',
    name: 'Mediation + Multi-Outcome',
    description:
      'Media drives awareness (mediator) feeding two outcomes with a cross-effect — resolves to CombinedMMM.',
    nodes: [
      { id: 'media_1', position: { x: 100, y: 100 }, data: { label: 'TV Spend', variableName: 'tv_spend', type: 'media' } },
      { id: 'media_2', position: { x: 300, y: 100 }, data: { label: 'Digital Spend', variableName: 'digital_spend', type: 'media' } },
      { id: 'mediator_1', position: { x: 200, y: 250 }, data: { label: 'Brand Awareness', variableName: 'awareness', type: 'mediator' } },
      { id: 'outcome_1', position: { x: 400, y: 400 }, data: { label: 'Revenue', variableName: 'revenue', type: 'outcome' } },
      { id: 'outcome_2', position: { x: 620, y: 400 }, data: { label: 'Volume', variableName: 'volume', type: 'outcome' } },
    ],
    edges: [
      { id: 'e1', source: 'media_1', target: 'mediator_1' },
      { id: 'e2', source: 'mediator_1', target: 'outcome_1' },
      { id: 'e3', source: 'mediator_1', target: 'outcome_2' },
      { id: 'e4', source: 'media_1', target: 'outcome_1' }, // Direct effect
      { id: 'e5', source: 'media_2', target: 'outcome_1' }, // Direct-only channel
      { id: 'e6', source: 'outcome_1', target: 'outcome_2' }, // Cross-effect
    ],
  },
];
