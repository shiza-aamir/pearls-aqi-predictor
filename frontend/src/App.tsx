import {
  lazy,
  Suspense,
  type ReactNode,
} from 'react'
import {
  createBrowserRouter,
  RouterProvider,
} from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'

const OverviewPage = lazy(() =>
  import('./pages/OverviewPage').then(
    (module) => ({
      default: module.OverviewPage,
    }),
  ),
)

const HistoryPage = lazy(() =>
  import('./pages/HistoryPage').then(
    (module) => ({
      default: module.HistoryPage,
    }),
  ),
)

const ModelInsightsPage = lazy(() =>
  import('./pages/ModelInsightsPage').then(
    (module) => ({
      default: module.ModelInsightsPage,
    }),
  ),
)

function PageFallback() {
  return (
    <div
      className="
        flex
        min-h-[240px]
        items-center
        justify-center
        text-[12px]
        text-[var(--color-text-tertiary)]
      "
    >
      Loading…
    </div>
  )
}

function withSuspense(
  element: ReactNode,
) {
  return (
    <Suspense fallback={<PageFallback />}>
      {element}
    </Suspense>
  )
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      {
        index: true,
        element: withSuspense(
          <OverviewPage />,
        ),
      },
      {
        path: 'history',
        element: withSuspense(
          <HistoryPage />,
        ),
      },
      {
        path: 'insights',
        element: withSuspense(
          <ModelInsightsPage />,
        ),
      },
    ],
  },
])

export default function App() {
  return (
    <RouterProvider router={router} />
  )
}