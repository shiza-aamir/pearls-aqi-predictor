import { apiGet } from './client'
import type { CitiesResponse } from './types'

export function getCities(
  signal?: AbortSignal,
): Promise<CitiesResponse> {
  return apiGet<CitiesResponse>(
    '/cities',
    signal,
  )
}