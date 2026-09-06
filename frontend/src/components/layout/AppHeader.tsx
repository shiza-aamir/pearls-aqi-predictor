import {
  Activity,
  Clock3,
  LayoutDashboard,
} from 'lucide-react'
import {
  NavLink,
} from 'react-router-dom'

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
    label: 'Model Insights',
    path: '/insights',
    icon: Activity,
  },
]

export function AppHeader() {
  return (
    <>
      <header
        className="
          sticky top-0 z-50
          hidden border-b
          border-[var(--color-border)]
          bg-[rgba(250,248,243,0.96)]
          md:block
        "
      >
        <div
          className="
            page-container
            flex h-[72px]
            items-center justify-between
          "
        >
          <NavLink
            to="/"
            className="flex items-baseline gap-3"
            aria-label="PEARLS overview"
          >
            <span
              className="
                font-display
                text-[24px]
                font-semibold
                tracking-[0.08em]
                text-[var(--color-text-primary)]
              "
            >
              PEARLS
            </span>

            <span
              className="
                hidden text-[12px]
                text-[var(--color-text-tertiary)]
                lg:inline
              "
            >
              Air Quality Index Prediction
            </span>
          </NavLink>

          <nav
            aria-label="Primary navigation"
            className="flex items-center gap-1"
          >
            {navigation.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  [
                    'rounded-[5px]',
                    'px-3 py-2',
                    'text-[13px]',
                    'font-medium',
                    'transition-colors',
                    isActive
                      ? 'bg-[var(--color-accent-soft)] text-[var(--color-accent-strong)]'
                      : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-sunken)] hover:text-[var(--color-text-primary)]',
                  ].join(' ')
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <header
        className="
          border-b
          border-[var(--color-border)]
          bg-[var(--color-background)]
          md:hidden
        "
      >
        <div
          className="
            page-container
            flex h-[64px]
            items-center
          "
        >
          <NavLink
            to="/"
            aria-label="PEARLS overview"
          >
            <span
              className="
                font-display
                text-[22px]
                font-semibold
                tracking-[0.08em]
              "
            >
              PEARLS
            </span>
          </NavLink>
        </div>
      </header>
    </>
  )
}