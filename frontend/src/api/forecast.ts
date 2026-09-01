import { apiGet } from './client'
import type { ForecastResponse } from './types'

export function getForecast(
  city: string,
  signal?: AbortSignal,
): Promise<ForecastResponse> {
  return apiGet<ForecastResponse>(
    `/forecast/${encodeURIComponent(city)}`,
    signal,
  )
}