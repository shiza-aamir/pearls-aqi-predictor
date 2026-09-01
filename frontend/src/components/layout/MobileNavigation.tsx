import {
  Activity,
  Clock3,
  LayoutDashboard,
} from 'lucide-react'
import { NavLink } from 'react-router-dom'

const navigation = [
  {
    label: 'Overview',
    path: '/',
    icon: LayoutDashboard,
  },
  {
    label: 'History',
    path: '/history',
    icon: Clock3,
  },
  {
    label: 'Insights',
    path: '/insights',
    icon: Activity,
  },
]

export function MobileNavigation() {
  return (
    <nav
      aria-label="Mobile navigation"
      className="
        fixed inset-x-0 bottom-0 z-50
        border-t
        border-[var(--color-border)]
        bg-[rgba(255,255,255,0.97)]
        md:hidden
      "
    >
      <div
        className="
          grid h-[68px]
          grid-cols-3
        "
      >
        {navigation.map((item) => {
          const Icon = item.icon

          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                [
                  'flex',
                  'flex-col',
                  'items-center',
                  'justify-center',
                  'gap-1',
                  'text-[10px]',
                  'font-medium',
                  isActive
                    ? 'text-[var(--color-accent-strong)]'
                    : 'text-[var(--color-text-tertiary)]',
                ].join(' ')
              }
            >
              <Icon
                size={18}
                strokeWidth={1.8}
                aria-hidden="true"
              />

              <span>{item.label}</span>
            </NavLink>
          )
        })}
      </div>
    </nav>
  )
}