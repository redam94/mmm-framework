import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PlanCompare } from './PlanCompare';
import type { BudgetPlanInfo } from '../../api/hooks/useBudgetPlans';

function plan(name: string, alloc: Record<string, number>): BudgetPlanInfo {
  return {
    plan_id: name,
    name,
    kind: 'optimization',
    created_at: 0,
    updated_at: 0,
    plan_payload: {
      allocation: Object.entries(alloc).map(([channel, optimal_spend]) => ({
        channel,
        optimal_spend,
      })),
    } as unknown as BudgetPlanInfo['plan_payload'],
  };
}

describe('PlanCompare', () => {
  it('shows per-channel spend for both plans and the delta', () => {
    const a = plan('A', { TV: 100, Search: 50 });
    const b = plan('B', { TV: 140, Search: 30 });
    render(<PlanCompare a={a} b={b} />);
    expect(screen.getByText('TV')).toBeInTheDocument();
    expect(screen.getByText('Search')).toBeInTheDocument();
    // delta column: TV +40, Search -20
    expect(screen.getByText('+40')).toBeInTheDocument();
    expect(screen.getByText('-20')).toBeInTheDocument();
  });

  it('handles a channel present in only one plan', () => {
    const a = plan('A', { TV: 100 });
    const b = plan('B', { TV: 100, Radio: 60 });
    render(<PlanCompare a={a} b={b} />);
    expect(screen.getByText('Radio')).toBeInTheDocument();
    expect(screen.getByText('+60')).toBeInTheDocument();
  });
});
