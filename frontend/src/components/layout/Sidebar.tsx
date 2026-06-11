import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';
import {
  HomeIcon,
  ChartBarIcon,
  ChatBubbleLeftRightIcon,
  CircleStackIcon,
  ArrowRightOnRectangleIcon,
} from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/authStore';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
}

const navigation: NavItem[] = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Runs', href: '/runs', icon: CircleStackIcon },
  { name: 'Chat', href: '/chat', icon: ChatBubbleLeftRightIcon },
];

export function Sidebar() {
  const location = useLocation();
  const { clearApiKey } = useAuthStore();

  const handleLogout = () => {
    clearApiKey();
  };

  return (
    <div className="flex h-full w-64 flex-col bg-gray-900">
      {/* Logo */}
      <div className="flex h-16 shrink-0 items-center px-6">
        <ChartBarIcon className="h-8 w-8 text-blue-500" />
        <span className="ml-3 text-xl font-bold text-white">MMM Studio</span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col px-4 py-4">
        <ul className="flex flex-1 flex-col gap-1">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href ||
              (item.href !== '/dashboard' && location.pathname.startsWith(item.href));

            return (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className={clsx(
                    'group flex items-center gap-x-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                  )}
                >
                  <item.icon
                    className={clsx(
                      'h-5 w-5 shrink-0',
                      isActive ? 'text-blue-500' : 'text-gray-400 group-hover:text-white'
                    )}
                  />
                  <span className="flex-1">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>

        {/* Logout */}
        <div className="mt-auto pt-4 border-t border-gray-800">
          <button
            onClick={handleLogout}
            className="flex w-full items-center gap-x-3 rounded-md px-3 py-2 text-sm font-medium text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
          >
            <ArrowRightOnRectangleIcon className="h-5 w-5 shrink-0" />
            <span>Sign Out</span>
          </button>
        </div>
      </nav>

      {/* Status footer */}
      <div className="border-t border-gray-800 px-4 py-3">
        <div className="flex items-center text-xs text-gray-500">
          <span className="h-2 w-2 rounded-full bg-green-500 mr-2" />
          <span>Connected</span>
        </div>
      </div>
    </div>
  );
}
