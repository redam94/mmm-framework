import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EnbsCard } from './EnbsCard';
import type { LearningRegret } from '../../api/services/learningService';

const BASE: LearningRegret = {
  e_regret_kpi: 3.2,
  e_regret_dollars: 41600,
  enbs: 16600,
  stop: false,
  margin: 1.0,
  population: 13,
  wave_cost: 25000,
};

describe('EnbsCard', () => {
  it('says to run the next wave while learning still pays (stop=false)', () => {
    render(<EnbsCard regret={BASE} />);
    expect(screen.getByText('Run next wave')).toBeInTheDocument();
    expect(screen.queryByText('Stop testing')).not.toBeInTheDocument();
    // E[regret] $ vs wave cost rows
    expect(screen.getByText('$41,600')).toBeInTheDocument();
    expect(screen.getByText('−$25,000')).toBeInTheDocument();
    expect(screen.getByText('$16,600')).toBeInTheDocument();
  });

  it('labels e_regret_kpi as value-$ per geo-period and population as geo-periods', () => {
    render(<EnbsCard regret={BASE} />);
    // e_regret_kpi already includes value_per_unit — never "KPI units"
    expect(
      screen.getByTitle(/expected value-\$ left on the table per geo-period/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/KPI\)/)).not.toBeInTheDocument();
    // the ENBS caption prices population in geo-periods (geos × horizon)
    expect(screen.getByText(/geo-periods = geos × horizon/)).toBeInTheDocument();
  });

  it('flips the verdict when the stopping rule fires (stop=true)', () => {
    render(
      <EnbsCard
        regret={{ ...BASE, stop: true, e_regret_dollars: 9000, enbs: -16000 }}
      />,
    );
    expect(screen.getByText('Stop testing')).toBeInTheDocument();
    expect(screen.queryByText('Run next wave')).not.toBeInTheDocument();
    expect(screen.getByText('−$16,000')).toBeInTheDocument();
  });
});
