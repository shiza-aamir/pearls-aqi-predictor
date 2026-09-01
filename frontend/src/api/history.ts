import { apiGet } from './client'
import type { HistoryResponse } from './types'

export type HistoryHours = 24 | 48 | 72 | 168

export function getHistory(
  city: string,
  hours: HistoryHours,
  signal?: AbortSignal,
): Promise<HistoryResponse> {
  return apiGet<HistoryResponse>(
    `/history/${encodeURIComponent(city)}?hours=${hours}`,
    signal,
  )
}