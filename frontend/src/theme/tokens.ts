export const colors = {
  background: '#FAF8F3',
  surface: '#FFFFFF',
  surfaceSunken: '#F3F0E8',

  border: '#E4DFD2',
  borderStrong: '#D3CDBC',

  textPrimary: '#1C2430',
  textSecondary: '#5B6472',
  textTertiary: '#8B93A0',

  accent: '#3F6D5C',
  accentSoft: '#E4EEE9',
  accentStrong: '#2C5245',

  aqi: {
    good: '#5B8A5A',
    moderate: '#B99A2E',
    sensitive: '#C97A3D',
    unhealthy: '#B54B3F',
    veryUnhealthy: '#7D4A82',
    hazardous: '#5C2E2C',
  },
} as const

export const typography = {
  display: '"Fraunces", Georgia, serif',
  body: '"Inter", system-ui, sans-serif',
  mono: '"IBM Plex Mono", monospace',
} as const

export const layout = {
  maxWidth: '1440px',
  contentWidth: '1240px',
  headerHeight: '72px',
  mobileNavigationHeight: '68px',
} as const

export const radius = {
  card: '6px',
  control: '5px',
  pill: '999px',
} as const