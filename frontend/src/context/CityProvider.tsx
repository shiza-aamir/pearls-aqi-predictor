import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { getCities } from '../api/cities'
import type {
  City,
} from '../api/types'
import {
  CITY_STORAGE_KEY,
  CityContext,
  DEFAULT_CITY,
} from './city-context'

interface CityProviderProps {
  children: ReactNode
}

function getStoredCity(): string {
  try {
    const storedCity =
      window.localStorage.getItem(
        CITY_STORAGE_KEY,
      )

    if (
      storedCity &&
      storedCity.trim()
    ) {
      return storedCity.trim()
    }
  } catch {
    // Storage unavailable.
  }

  return DEFAULT_CITY
}

export function CityProvider({
  children,
}: CityProviderProps) {
  const [
    selectedCity,
    setSelectedCityState,
  ] = useState(getStoredCity)

  const [cities, setCities] =
    useState<City[]>([])

  const [
    citiesLoading,
    setCitiesLoading,
  ] = useState(true)

  const [
    citiesError,
    setCitiesError,
  ] = useState<string | null>(null)

  useEffect(() => {
    const controller =
      new AbortController()

    async function loadCities() {
      try {
        setCitiesLoading(true)
        setCitiesError(null)

        const response =
          await getCities(
            controller.signal,
          )

        if (
          controller.signal.aborted
        ) {
          return
        }

        setCities(response.cities)

        const storedSelectionSupported =
          response.cities.some(
            (city) =>
              city.name === selectedCity,
          )

        if (!storedSelectionSupported) {
          setSelectedCityState(
            DEFAULT_CITY,
          )

          try {
            window.localStorage.setItem(
              CITY_STORAGE_KEY,
              DEFAULT_CITY,
            )
          } catch {
            // Persistence is optional.
          }
        }
      } catch (requestError) {
        if (
          requestError instanceof
            DOMException &&
          requestError.name ===
            'AbortError'
        ) {
          return
        }

        setCitiesError(
          requestError instanceof Error
            ? requestError.message
            : 'Unable to load supported cities.',
        )
      } finally {
        if (
          !controller.signal.aborted
        ) {
          setCitiesLoading(false)
        }
      }
    }

    void loadCities()

    return () => {
      controller.abort()
    }

    // Intentionally runs once.
    // The supported-city catalogue is
    // application-level reference data.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const setSelectedCity =
    useCallback(
      (city: string) => {
        const normalizedCity =
          city.trim()

        if (
          !normalizedCity ||
          normalizedCity ===
            selectedCity
        ) {
          return
        }

        setSelectedCityState(
          normalizedCity,
        )

        try {
          window.localStorage.setItem(
            CITY_STORAGE_KEY,
            normalizedCity,
          )
        } catch {
          // Selection still works in memory.
        }
      },
      [selectedCity],
    )

  const value = useMemo(
    () => ({
      cities,
      citiesLoading,
      citiesError,

      selectedCity,
      setSelectedCity,
    }),
    [
      cities,
      citiesLoading,
      citiesError,
      selectedCity,
      setSelectedCity,
    ],
  )

  return (
    <CityContext.Provider value={value}>
      {children}
    </CityContext.Provider>
  )
}