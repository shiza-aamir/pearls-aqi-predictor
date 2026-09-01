import { apiGet } from './client'
import type { ModelsResponse } from './types'

export function getModels(
  signal?: AbortSignal,
): Promise<ModelsResponse> {
  return apiGet<ModelsResponse>(
    '/models',
    signal,
  )
}