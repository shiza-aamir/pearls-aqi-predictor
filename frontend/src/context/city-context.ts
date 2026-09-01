import {
  createContext,
} from 'react'

import type {
  City,
} from '../api/types'

export const DEFAULT_CITY = 'Islamabad'

export const CITY_STORAGE_KEY =
  'pearls-selected-city'

export interface CityContextValue {
  cities: City[]
  citiesLoading: boolean
  citiesError: string | null

  selectedCity: string
  setSelectedCity: (city: string) => void
}

export const CityContext =
  createContext<CityContextValue | undefined>(
    undefined,
  )