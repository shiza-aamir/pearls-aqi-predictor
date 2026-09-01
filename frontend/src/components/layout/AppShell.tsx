import { Outlet } from 'react-router-dom'

import { AppHeader } from './AppHeader'
import { MobileNavigation } from './MobileNavigation'
import { ScrollRestoration } from './ScrollRestoration'

export function AppShell() {
  return (
    <div className="min-h-screen">
      <ScrollRestoration />

      <AppHeader />

      <main
        className="
          pb-[92px]
          md:pb-0
        "
      >
        <Outlet />
      </main>

      <MobileNavigation />
    </div>
  )
}