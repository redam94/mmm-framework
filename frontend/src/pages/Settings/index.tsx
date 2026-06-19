import { useState } from 'react';
import { SectionHeader, Tabs } from '../../components/ui';
import { ProfileSection } from './ProfileSection';
import { SecuritySection } from './SecuritySection';
import { ModelSection } from './ModelSection';
import { DataConnectionsSection } from './DataConnectionsSection';

type TabId = 'profile' | 'security' | 'model' | 'connections';

const TABS = [
  { id: 'profile', label: 'Profile' },
  { id: 'security', label: 'Security' },
  { id: 'model', label: 'Model & API' },
  { id: 'connections', label: 'Data connections' },
];

export function SettingsPage() {
  const [active, setActive] = useState<TabId>('profile');

  return (
    <div className="space-y-6">
      <SectionHeader
        level={1}
        title="Settings"
        subtitle="Your account, security, the model the agent runs on, and connected data sources."
      />
      <Tabs tabs={TABS} active={active} onChange={(id) => setActive(id as TabId)} />

      {active === 'profile' && <ProfileSection />}
      {active === 'security' && <SecuritySection />}
      {active === 'model' && <ModelSection />}
      {active === 'connections' && <DataConnectionsSection />}
    </div>
  );
}
