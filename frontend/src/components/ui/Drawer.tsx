import { Fragment, type ReactNode } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { X } from 'lucide-react';

interface DrawerProps {
  open: boolean;
  onClose: () => void;
  title?: ReactNode;
  children: ReactNode;
  /** Tailwind max-width class for the panel */
  width?: string;
}

/** Right slide-over panel (deep-linkable detail views). */
export function Drawer({ open, onClose, title, children, width = 'max-w-xl' }: DrawerProps) {
  return (
    <Transition.Root show={open} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-in-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in-out duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-ink-900/40 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-300"
                enterFrom="translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-200"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel className={`pointer-events-auto w-screen ${width}`}>
                  <div className="flex h-full flex-col overflow-y-auto bg-cream-50 shadow-xl scrollbar-thin">
                    <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line-200 bg-cream-50/95 px-6 py-4 backdrop-blur">
                      <Dialog.Title className="font-display text-lg font-semibold text-ink-900">
                        {title}
                      </Dialog.Title>
                      <button
                        type="button"
                        className="rounded-md p-1 text-ink-400 transition-colors hover:bg-cream-200 hover:text-ink-900"
                        onClick={onClose}
                      >
                        <span className="sr-only">Close panel</span>
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                    <div className="flex-1 px-6 py-5">{children}</div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
}
