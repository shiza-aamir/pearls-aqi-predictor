import { apiGet } from './client'
import type { PerformanceResponse } from './types'

export function getPerformance(
  city: string,
  signal?: AbortSignal,
): Promise<PerformanceResponse> {
  return apiGet<PerformanceResponse>(
    `/performance/${encodeURIComponent(city)}`,
    signal,
  )
}