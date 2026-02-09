import { useLocation } from 'react-router-dom';
import { Bars3Icon } from '@heroicons/react/24/outline';

interface HeaderProps {
  onMenuClick?: () => void;
}

// Page titles mapping
const PAGE_TITLES: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/planning': 'Model Planning',
  '/data': 'Data Management',
  '/config': 'Model Configuration',
  '/fit': 'Model Fitting',
  '/diagnostics': 'Model Diagnostics',
  '/results': 'Results & Export',
};

export function Header({ onMenuClick }: HeaderProps) {
  const location = useLocation();

  // Get page title from path
  const getPageTitle = () => {
    // Exact match first
    if (PAGE_TITLES[location.pathname]) {
      return PAGE_TITLES[location.pathname];
    }
    // Check for partial matches (e.g., /models/123)
    for (const [path, title] of Object.entries(PAGE_TITLES)) {
      if (location.pathname.startsWith(path)) {
        return title;
      }
    }
    return 'MMM Studio';
  };

  return (
    <header className="sticky top-0 z-10 flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
      {/* Mobile menu button */}
      <button
        type="button"
        className="-m-2.5 p-2.5 text-gray-700 lg:hidden"
        onClick={onMenuClick}
      >
        <span className="sr-only">Open sidebar</span>
        <Bars3Icon className="h-6 w-6" aria-hidden="true" />
      </button>

      {/* Separator */}
      <div className="h-6 w-px bg-gray-200 lg:hidden" aria-hidden="true" />

      {/* Page title */}
      <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
        <div className="flex items-center">
          <h1 className="text-xl font-semibold text-gray-900">{getPageTitle()}</h1>
        </div>

        {/* Right side actions could go here */}
        <div className="flex flex-1 items-center justify-end gap-x-4 lg:gap-x-6">
          {/* Placeholder for notifications, user menu, etc. */}
        </div>
      </div>
    </header>
  );
}
