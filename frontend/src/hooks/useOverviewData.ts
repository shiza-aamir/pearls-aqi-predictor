import {
  useCallback,
  useEffect,
  useState,
} from 'react'

import { getForecast } from '../api/forecast'
import { getPerformance } from '../api/performance'
import type {
  City,
  ForecastResponse,
  PerformanceResponse,
} from '../api/types'
import { useCity } from './useCity'

export interface OverviewDataState {
  cities: City[]
  selectedCity: string

  forecast: ForecastResponse | null
  performance: PerformanceResponse | null

  citiesLoading: boolean
  forecastLoading: boolean
  performanceLoading: boolean

  forecastError: string | null
  performanceError: string | null

  selectCity: (city: string) => void
  refresh: () => void
}

export function useOverviewData():
  OverviewDataState {
  const {
    cities,
    citiesLoading,
    citiesError,
    selectedCity,
    setSelectedCity,
  } = useCity()

  const [forecast, setForecast] =
    useState<ForecastResponse | null>(null)

  const [performance, setPerformance] =
    useState<PerformanceResponse | null>(
      null,
    )

  const [
    forecastLoading,
    setForecastLoading,
  ] = useState(true)

  const [
    performanceLoading,
    setPerformanceLoading,
  ] = useState(true)

  const [
    forecastError,
    setForecastError,
  ] = useState<string | null>(null)

  const [
    performanceError,
    setPerformanceError,
  ] = useState<string | null>(null)

  const [refreshKey, setRefreshKey] =
    useState(0)

  useEffect(() => {
    const controller =
      new AbortController()

    async function loadForecast() {
      try {
        setForecastLoading(true)
        setForecastError(null)

        const response =
          await getForecast(
            selectedCity,
            controller.signal,
          )

        setForecast(response)
      } catch (requestError) {
        if (
          requestError instanceof
            DOMException &&
          requestError.name ===
            'AbortError'
        ) {
          return
        }

        setForecastError(
          requestError instanceof Error
            ? requestError.message
            : 'Unable to load the latest forecast.',
        )
      } finally {
        if (
          !controller.signal.aborted
        ) {
          setForecastLoading(false)
        }
      }
    }

    void loadForecast()

    return () => {
      controller.abort()
    }
  }, [
    selectedCity,
    refreshKey,
  ])

  useEffect(() => {
    const controller =
      new AbortController()

    async function loadPerformance() {
      try {
        setPerformanceLoading(true)
        setPerformanceError(null)

        const response =
          await getPerformance(
            selectedCity,
            controller.signal,
          )

        setPerformance(response)
      } catch (requestError) {
        if (
          requestError instanceof
            DOMException &&
          requestError.name ===
            'AbortError'
        ) {
          return
        }

        setPerformanceError(
          requestError instanceof Error
            ? requestError.message
            : 'Unable to load forecast performance.',
        )
      } finally {
        if (
          !controller.signal.aborted
        ) {
          setPerformanceLoading(false)
        }
      }
    }

    void loadPerformance()

    return () => {
      controller.abort()
    }
  }, [
    selectedCity,
    refreshKey,
  ])

  const selectCity = useCallback(
    (city: string) => {
      if (city === selectedCity) {
        return
      }

      setForecast(null)
      setPerformance(null)
      setSelectedCity(city)
    },
    [
      selectedCity,
      setSelectedCity,
    ],
  )

  const refresh = useCallback(() => {
    setRefreshKey(
      (current) => current + 1,
    )
  }, [])

  return {
    cities,
    selectedCity,

    forecast,
    performance,

    citiesLoading,
    forecastLoading,
    performanceLoading,

    forecastError:
      forecastError ?? citiesError,

    performanceError,

    selectCity,
    refresh,
  }
}