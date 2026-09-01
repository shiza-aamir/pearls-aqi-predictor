import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'
import { CityProvider } from './context/CityProvider'
import './index.css'

const rootElement =
  document.getElementById('root')

if (!rootElement) {
  throw new Error(
    'Root element was not found.',
  )
}

createRoot(rootElement).render(
  <StrictMode>
    <CityProvider>
      <App />
    </CityProvider>
  </StrictMode>,
)