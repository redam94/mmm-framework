import { useState } from 'react';
import { Button, TextInput, Textarea, Card, Title, Text, Badge } from '@tremor/react';
import { PencilSquareIcon, CheckIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useWorkflowStore, type BayesianPhase } from '../../stores/workflowStore';

interface RationaleLoggerProps {
  phase: BayesianPhase;
  decisionType: string;
  title?: string;
  description?: string;
  minLength?: number;
  className?: string;
}

export function RationaleLogger({
  phase,
  decisionType,
  title = 'Document Decision',
  description,
  minLength = 50,
  className,
}: RationaleLoggerProps) {
  const { logDecision, decisions } = useWorkflowStore();
  const [isOpen, setIsOpen] = useState(false);
  const [rationale, setRationale] = useState('');

  const existingDecisions = decisions.filter(
    (d) => d.phase === phase && d.type === decisionType
  );

  const handleSave = () => {
    if (rationale.length >= minLength) {
      logDecision({
        phase,
        type: decisionType,
        rationale,
      });
      setRationale('');
      setIsOpen(false);
    }
  };

  const isValid = rationale.length >= minLength;

  return (
    <div className={className}>
      {/* Existing decisions */}
      {existingDecisions.length > 0 && (
        <div className="mb-4 space-y-2">
          <Text className="text-sm font-medium text-gray-700">Previous decisions:</Text>
          {existingDecisions.map((decision, idx) => (
            <div key={idx} className="p-3 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex justify-between items-start">
                <Text className="text-sm">{decision.rationale}</Text>
                <Badge color="gray" size="xs">
                  {new Date(decision.timestamp).toLocaleString()}
                </Badge>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Add new decision */}
      {!isOpen ? (
        <Button
          icon={PencilSquareIcon}
          variant="secondary"
          onClick={() => setIsOpen(true)}
          size="sm"
        >
          {existingDecisions.length > 0 ? 'Add Another Decision' : title}
        </Button>
      ) : (
        <Card className="p-4">
          <Title className="text-sm">{title}</Title>
          {description && <Text className="text-xs text-gray-500 mt-1">{description}</Text>}

          <Textarea
            className="mt-3"
            placeholder={`Enter your rationale (minimum ${minLength} characters)...`}
            value={rationale}
            onChange={(e) => setRationale(e.target.value)}
            rows={4}
          />

          <div className="mt-2 flex justify-between items-center">
            <Text
              className={`text-xs ${isValid ? 'text-green-600' : 'text-gray-400'}`}
            >
              {rationale.length}/{minLength} characters
            </Text>
            <div className="flex gap-2">
              <Button
                icon={XMarkIcon}
                variant="secondary"
                size="xs"
                onClick={() => {
                  setRationale('');
                  setIsOpen(false);
                }}
              >
                Cancel
              </Button>
              <Button
                icon={CheckIcon}
                size="xs"
                onClick={handleSave}
                disabled={!isValid}
              >
                Save
              </Button>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

// Compact inline rationale input
interface InlineRationaleProps {
  placeholder?: string;
  minLength?: number;
  onSave: (rationale: string) => void;
  className?: string;
}

export function InlineRationale({
  placeholder = 'Why are you making this change?',
  minLength = 20,
  onSave,
  className,
}: InlineRationaleProps) {
  const [value, setValue] = useState('');
  const isValid = value.length >= minLength;

  const handleSubmit = () => {
    if (isValid) {
      onSave(value);
      setValue('');
    }
  };

  return (
    <div className={`flex gap-2 ${className}`}>
      <TextInput
        placeholder={placeholder}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="flex-1"
      />
      <Button size="xs" onClick={handleSubmit} disabled={!isValid}>
        Save
      </Button>
    </div>
  );
}
