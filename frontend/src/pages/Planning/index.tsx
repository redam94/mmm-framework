import { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  BackgroundVariant,
} from '@xyflow/react';
import type { Connection, Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Card, Title, Text, Button, Select, SelectItem } from '@tremor/react';
import { PlayIcon, TrashIcon, DocumentTextIcon } from '@heroicons/react/24/outline';
import { useDAGStore, type DAGNode, type DAGNodeData, type DAGNodeType } from '../../stores/dagStore';
import { useWorkflowStore } from '../../stores/workflowStore';

// Node type configurations
const NODE_TYPES: { type: DAGNodeType; label: string; color: string }[] = [
  { type: 'kpi', label: 'KPI/Outcome', color: '#10B981' },
  { type: 'media', label: 'Media Channel', color: '#3B82F6' },
  { type: 'control', label: 'Control Variable', color: '#8B5CF6' },
  { type: 'mediator', label: 'Mediator', color: '#F59E0B' },
];

// Custom node component
function CustomNode({ data }: { data: DAGNodeData }) {
  const colorMap: Record<DAGNodeType, string> = {
    kpi: 'bg-green-100 border-green-500',
    media: 'bg-blue-100 border-blue-500',
    control: 'bg-purple-100 border-purple-500',
    mediator: 'bg-yellow-100 border-yellow-500',
    transform: 'bg-gray-100 border-gray-500',
    outcome: 'bg-green-100 border-green-500',
  };

  return (
    <div
      className={`px-4 py-2 rounded-lg border-2 shadow-sm ${colorMap[data.type] || 'bg-gray-100 border-gray-300'}`}
    >
      <div className="text-sm font-medium">{data.label}</div>
      <div className="text-xs text-gray-500">{data.type}</div>
    </div>
  );
}

const nodeTypes = {
  default: CustomNode,
};

export function PlanningPage() {
  const {
    nodes: storeNodes,
    edges: storeEdges,
    addNode,
    setNodes: setStoreNodes,
    setEdges: setStoreEdges,
    clearDAG,
    validateDAG,
    generateNarrative,
    loadTemplate,
    selectedNodeId,
    setSelectedNode,
  } = useDAGStore();

  const { logDecision } = useWorkflowStore();

  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState(storeNodes as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges);

  // Sync with store
  const syncToStore = useCallback(() => {
    setStoreNodes(nodes as DAGNode[]);
    setStoreEdges(edges);
  }, [nodes, edges, setStoreNodes, setStoreEdges]);

  // Handle new connections
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges]
  );

  // Handle node selection
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNode(node.id);
    },
    [setSelectedNode]
  );

  // Add new node
  const handleAddNode = (type: DAGNodeType) => {
    const newNode: DAGNode = {
      id: `node_${Date.now()}`,
      type: 'default',
      position: { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 },
      data: {
        label: `New ${type}`,
        type,
        variableName: `var_${Date.now()}`,
        config: {},
      },
    };
    addNode(newNode);
    setNodes((nds) => [...nds, newNode as Node]);
  };

  // Validate and log decision
  const handleValidateAndContinue = () => {
    syncToStore();
    const validation = validateDAG();

    if (validation.valid) {
      logDecision({
        phase: 'planning',
        type: 'dag_complete',
        rationale: `Defined generative model with ${nodes.length} nodes and ${edges.length} edges. ${generateNarrative()}`,
      });
      alert('DAG is valid! You can proceed to the next phase.');
    } else {
      alert(`Validation errors:\n${validation.errors.join('\n')}`);
    }
  };

  // Load template
  const handleLoadTemplate = (template: 'simple' | 'mediation' | 'multivariate') => {
    loadTemplate(template);
    // Reload nodes from store
    const store = useDAGStore.getState();
    setNodes(store.nodes as Node[]);
    setEdges(store.edges);
  };

  // Narrative
  const narrative = useMemo(() => generateNarrative(), [nodes, edges]);

  // Selected node details
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <Title>Model Planning</Title>
          <Text>Design your data generative story with a DAG</Text>
        </div>
        <div className="flex gap-2">
          <Select
            placeholder="Load template"
            onValueChange={(v) => handleLoadTemplate(v as 'simple' | 'mediation' | 'multivariate')}
          >
            <SelectItem value="simple">Simple MMM</SelectItem>
            <SelectItem value="mediation">With Mediation</SelectItem>
            <SelectItem value="multivariate">Multi-Outcome</SelectItem>
          </Select>
          <Button
            icon={TrashIcon}
            variant="secondary"
            onClick={() => {
              clearDAG();
              setNodes([]);
              setEdges([]);
            }}
          >
            Clear
          </Button>
          <Button icon={PlayIcon} onClick={handleValidateAndContinue}>
            Validate & Continue
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-4 gap-6">
        {/* Node palette */}
        <Card className="col-span-1">
          <Title className="text-sm">Add Nodes</Title>
          <div className="mt-4 space-y-2">
            {NODE_TYPES.map(({ type, label, color }) => (
              <button
                key={type}
                onClick={() => handleAddNode(type)}
                className="w-full p-3 text-left rounded-lg border-2 border-dashed hover:border-solid transition-colors"
                style={{ borderColor: color, backgroundColor: `${color}20` }}
              >
                <span className="text-sm font-medium">{label}</span>
              </button>
            ))}
          </div>

          {/* Selected node editor */}
          {selectedNode && (
            <div className="mt-6 pt-6 border-t">
              <Title className="text-sm">Selected: {(selectedNode.data as DAGNodeData).label}</Title>
              <div className="mt-4 space-y-2">
                <div>
                  <label className="text-xs text-gray-500">Variable Name</label>
                  <input
                    type="text"
                    className="w-full mt-1 px-2 py-1 text-sm border rounded"
                    value={(selectedNode.data as DAGNodeData).variableName || ''}
                    onChange={(e) => {
                      setNodes((nds) =>
                        nds.map((n) =>
                          n.id === selectedNode.id
                            ? { ...n, data: { ...n.data, variableName: e.target.value } }
                            : n
                        )
                      );
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* DAG Canvas */}
        <Card className="col-span-2 h-[600px]">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
          >
            <Controls />
            <MiniMap />
            <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          </ReactFlow>
        </Card>

        {/* Narrative panel */}
        <Card className="col-span-1">
          <div className="flex items-center gap-2">
            <DocumentTextIcon className="h-5 w-5 text-gray-500" />
            <Title className="text-sm">Generated Narrative</Title>
          </div>
          <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <Text className="text-sm">
              {narrative || 'Add nodes to the canvas to generate a narrative.'}
            </Text>
          </div>

          {/* Validation status */}
          <div className="mt-6">
            <Title className="text-sm">Validation</Title>
            <div className="mt-2 space-y-1">
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${nodes.length > 0 ? 'bg-green-500' : 'bg-gray-300'}`}
                />
                <Text className="text-xs">Has nodes</Text>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${edges.length > 0 ? 'bg-green-500' : 'bg-gray-300'}`}
                />
                <Text className="text-xs">Has connections</Text>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${nodes.some((n) => (n.data as DAGNodeData).type === 'kpi') ? 'bg-green-500' : 'bg-gray-300'}`}
                />
                <Text className="text-xs">Has KPI/Outcome</Text>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${nodes.some((n) => (n.data as DAGNodeData).type === 'media') ? 'bg-green-500' : 'bg-gray-300'}`}
                />
                <Text className="text-xs">Has media channels</Text>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default PlanningPage;
