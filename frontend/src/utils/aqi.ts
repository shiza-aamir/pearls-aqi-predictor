export type AQICategory =
  | 'Good'
  | 'Moderate'
  | 'Unhealthy for Sensitive Groups'
  | 'Unhealthy'
  | 'Very Unhealthy'
  | 'Hazardous'

export interface AQICategoryPresentation {
  label: string
  shortLabel: string
  color: string
}

const CATEGORY_PRESENTATION: Record<
  AQICategory,
  AQICategoryPresentation
> = {
  Good: {
    label: 'Good',
    shortLabel: 'Good',
    color: '#5B8A5A',
  },

  Moderate: {
    label: 'Moderate',
    shortLabel: 'Moderate',
    color: '#B99A2E',
  },

  'Unhealthy for Sensitive Groups': {
    label: 'Unhealthy for Sensitive Groups',
    shortLabel: 'Sensitive Groups',
    color: '#C97A3D',
  },

  Unhealthy: {
    label: 'Unhealthy',
    shortLabel: 'Unhealthy',
    color: '#B54B3F',
  },

  'Very Unhealthy': {
    label: 'Very Unhealthy',
    shortLabel: 'Very Unhealthy',
    color: '#7D4A82',
  },

  Hazardous: {
    label: 'Hazardous',
    shortLabel: 'Hazardous',
    color: '#5C2E2C',
  },
}

export function getAQICategoryPresentation(
  category: string,
): AQICategoryPresentation {
  if (category in CATEGORY_PRESENTATION) {
    return CATEGORY_PRESENTATION[
      category as AQICategory
    ]
  }

  return {
    label: category,
    shortLabel: category,
    color: '#5B6472',
  }
}