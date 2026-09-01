import {
  useCallback,
  useEffect,
  useState,
} from 'react'

import {
  getHistory,
  type HistoryHours,
} from '../api/history'
import type {
  City,
  HistoryResponse,
} from '../api/types'
import { useCity } from './useCity'

const DEFAULT_HOURS: HistoryHours = 24

export interface HistoryDataState {
  cities: City[]
  selectedCity: string
  selectedHours: HistoryHours

  history: HistoryResponse | null

  citiesLoading: boolean
  historyLoading: boolean

  error: string | null

  selectCity: (city: string) => void
  selectHours: (
    hours: HistoryHours,
  ) => void
  refresh: () => void
}

export function useHistoryData():
  HistoryDataState {
  const {
    cities,
    citiesLoading,
    citiesError,
    selectedCity,
    setSelectedCity,
  } = useCity()

  const [
    selectedHours,
    setSelectedHours,
  ] =
    useState<HistoryHours>(
      DEFAULT_HOURS,
    )

  const [history, setHistory] =
    useState<HistoryResponse | null>(
      null,
    )

  const [
    historyLoading,
    setHistoryLoading,
  ] = useState(true)

  const [error, setError] =
    useState<string | null>(null)

  const [refreshKey, setRefreshKey] =
    useState(0)

  useEffect(() => {
    const controller =
      new AbortController()

    async function loadHistory() {
      try {
        setHistoryLoading(true)
        setError(null)

        const response =
          await getHistory(
            selectedCity,
            selectedHours,
            controller.signal,
          )

        setHistory(response)
      } catch (requestError) {
        if (
          requestError instanceof
            DOMException &&
          requestError.name ===
            'AbortError'
        ) {
          return
        }

        setError(
          requestError instanceof Error
            ? requestError.message
            : 'Unable to load historical air quality data.',
        )
      } finally {
        if (
          !controller.signal.aborted
        ) {
          setHistoryLoading(false)
        }
      }
    }

    void loadHistory()

    return () => {
      controller.abort()
    }
  }, [
    selectedCity,
    selectedHours,
    refreshKey,
  ])

  const selectCity = useCallback(
    (city: string) => {
      if (city === selectedCity) {
        return
      }

      setHistory(null)
      setSelectedCity(city)
    },
    [
      selectedCity,
      setSelectedCity,
    ],
  )

  const selectHours = useCallback(
    (hours: HistoryHours) => {
      if (hours === selectedHours) {
        return
      }

      setSelectedHours(hours)
      setHistory(null)
    },
    [selectedHours],
  )

  const refresh = useCallback(() => {
    setRefreshKey(
      (current) => current + 1,
    )
  }, [])

  return {
    cities,
    selectedCity,
    selectedHours,

    history,

    citiesLoading,
    historyLoading,

    error: error ?? citiesError,

    selectCity,
    selectHours,
    refresh,
  }
}