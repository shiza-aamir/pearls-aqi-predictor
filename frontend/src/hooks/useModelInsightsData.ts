import {
  useCallback,
  useEffect,
  useState,
} from 'react'

import { getForecast } from '../api/forecast'
import { getModels } from '../api/models'
import { getPerformance } from '../api/performance'
import type {
  City,
  ForecastResponse,
  ModelsResponse,
  PerformanceResponse,
} from '../api/types'
import { useCity } from './useCity'

export interface ModelInsightsDataState {
  cities: City[]
  selectedCity: string

  models: ModelsResponse | null
  performance: PerformanceResponse | null
  forecast: ForecastResponse | null

  citiesLoading: boolean
  modelsLoading: boolean
  performanceLoading: boolean
  forecastLoading: boolean

  modelsError: string | null
  performanceError: string | null
  forecastError: string | null

  selectCity: (city: string) => void
  refresh: () => void
}

export function useModelInsightsData():
  ModelInsightsDataState {
  const {
    cities,
    citiesLoading,
    citiesError,
    selectedCity,
    setSelectedCity,
  } = useCity()

  const [models, setModels] =
    useState<ModelsResponse | null>(null)

  const [performance, setPerformance] =
    useState<PerformanceResponse | null>(
      null,
    )

  const [forecast, setForecast] =
    useState<ForecastResponse | null>(null)

  const [
    modelsLoading,
    setModelsLoading,
  ] = useState(true)

  const [
    performanceLoading,
    setPerformanceLoading,
  ] = useState(true)

  const [
    forecastLoading,
    setForecastLoading,
  ] = useState(true)

  const [
    modelsError,
    setModelsError,
  ] = useState<string | null>(null)

  const [
    performanceError,
    setPerformanceError,
  ] = useState<string | null>(null)

  const [
    forecastError,
    setForecastError,
  ] = useState<string | null>(null)

  const [refreshKey, setRefreshKey] =
    useState(0)

  useEffect(() => {
    const controller =
      new AbortController()

    async function loadModels() {
      try {
        setModelsLoading(true)
        setModelsError(null)

        const response =
          await getModels(
            controller.signal,
          )

        setModels(response)
      } catch (requestError) {
        if (
          requestError instanceof
            DOMException &&
          requestError.name ===
            'AbortError'
        ) {
          return
        }

        setModelsError(
          requestError instanceof Error
            ? requestError.message
            : 'Unable to load model information.',
        )
      } finally {
        if (
          !controller.signal.aborted
        ) {
          setModelsLoading(false)
        }
      }
    }

    void loadModels()

    return () => {
      controller.abort()
    }
  }, [refreshKey])

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
            : 'Unable to load model performance.',
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
            : 'Unable to load forecast explanations.',
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

  const selectCity = useCallback(
    (city: string) => {
      if (city === selectedCity) {
        return
      }

      setPerformance(null)
      setForecast(null)
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

    models,
    performance,
    forecast,

    citiesLoading,
    modelsLoading,
    performanceLoading,
    forecastLoading,

    modelsError:
      modelsError ?? citiesError,

    performanceError,
    forecastError,

    selectCity,
    refresh,
  }
}