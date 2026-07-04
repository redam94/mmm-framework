import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { OffPanelFields, emptyOffPanel, offPanelReadoutFields } from './OffPanelFields';

describe('offPanelReadoutFields', () => {
  it('is empty until a spend level is entered', () => {
    expect(offPanelReadoutFields(emptyOffPanel)).toEqual({});
    expect(
      offPanelReadoutFields({ ...emptyOffPanel, treatedUnits: '5', adstockState: 'cold_start' }),
    ).toEqual({});
  });

  it('keeps the spend delta SIGNED (a holdout is negative)', () => {
    expect(offPanelReadoutFields({ ...emptyOffPanel, spendPerPeriod: '-5000' })).toEqual({
      spend_per_period: -5000,
      n_treated_units: 1,
      adstock_state: 'steady_state',
    });
  });

  it('carries treated units and adstock state alongside the spend level', () => {
    expect(
      offPanelReadoutFields({
        spendPerPeriod: '2500',
        treatedUnits: '4',
        adstockState: 'cold_start',
      }),
    ).toEqual({
      spend_per_period: 2500,
      n_treated_units: 4,
      adstock_state: 'cold_start',
    });
  });

  it('floors treated units at 1', () => {
    expect(
      offPanelReadoutFields({ ...emptyOffPanel, spendPerPeriod: '100', treatedUnits: '0' }),
    ).toMatchObject({ n_treated_units: 1 });
  });

  it('drops non-numeric spend input', () => {
    expect(offPanelReadoutFields({ ...emptyOffPanel, spendPerPeriod: 'abc' })).toEqual({});
  });
});

describe('OffPanelFields', () => {
  it('starts collapsed and reveals the three inputs on toggle', () => {
    render(<OffPanelFields state={emptyOffPanel} onChange={() => {}} inputCls="" />);
    expect(screen.queryByText(/Spend Δ \/ period/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Off-panel calibration'));
    expect(screen.getByText(/Spend Δ \/ period/)).toBeInTheDocument();
    expect(screen.getByText('Treated units')).toBeInTheDocument();
    expect(screen.getByText('Adstock state')).toBeInTheDocument();
  });

  it('starts open when a spend level is already set', () => {
    render(
      <OffPanelFields
        state={{ ...emptyOffPanel, spendPerPeriod: '-100' }}
        onChange={() => {}}
        inputCls=""
      />,
    );
    expect(screen.getByText(/Spend Δ \/ period/)).toBeInTheDocument();
  });

  it('propagates edits through onChange', () => {
    const onChange = vi.fn();
    render(
      <OffPanelFields
        state={{ ...emptyOffPanel, spendPerPeriod: '-100' }}
        onChange={onChange}
        inputCls=""
      />,
    );
    fireEvent.change(screen.getByPlaceholderText('1'), { target: { value: '3' } });
    expect(onChange).toHaveBeenCalledWith({
      ...emptyOffPanel,
      spendPerPeriod: '-100',
      treatedUnits: '3',
    });
  });
});
