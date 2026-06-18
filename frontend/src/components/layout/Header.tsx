import { useLocation } from 'react-router-dom';
import { Menu } from 'lucide-react';
import { APP_NAME, pageForPath } from '../../appIdentity';
import { ModelSwitcher } from '../common';
import { ProjectSwitcher } from '../common/ProjectSwitcher';

interface HeaderProps {
  onMenuClick?: () => void;
}

export function Header({ onMenuClick }: HeaderProps) {
  const location = useLocation();
  const page = pageForPath(location.pathname);

  // z-30: above the workspace tab bar (sticky z-10) so the project/model
  // dropdowns overlay the tabs, and below the sidebar/modals (z-50).
  return (
    <header className="sticky top-0 z-30 flex h-16 shrink-0 items-center gap-x-4 border-b border-line-200 bg-cream-50/90 px-4 backdrop-blur sm:gap-x-6 sm:px-6 lg:px-8">
      {/* Mobile menu button */}
      <button
        type="button"
        className="-m-2.5 p-2.5 text-ink-700 lg:hidden"
        onClick={onMenuClick}
      >
        <span className="sr-only">Open sidebar</span>
        <Menu className="h-6 w-6" aria-hidden="true" />
      </button>

      {/* Separator */}
      <div className="h-6 w-px bg-line-300 lg:hidden" aria-hidden="true" />

      {/* Page title */}
      <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
        <div className="flex items-baseline gap-3">
          <h1 className="font-display text-3xl font-semibold tracking-tight text-ink-900">
            {page?.name ?? APP_NAME}
          </h1>
          {page && (
            <span className="hidden text-sm text-ink-400 sm:inline">{page.hint}</span>
          )}
        </div>

        <div className="flex flex-1 items-center justify-end gap-x-3 lg:gap-x-4">
          <ProjectSwitcher />
          <ModelSwitcher theme="light" />
        </div>
      </div>
    </header>
  );
}
