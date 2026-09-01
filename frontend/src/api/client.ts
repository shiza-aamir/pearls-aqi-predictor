const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  'http://127.0.0.1:8000/api/v1'

export class APIError extends Error {
  readonly status: number

  constructor(
    message: string,
    status: number,
  ) {
    super(message)

    this.name = 'APIError'
    this.status = status
  }
}

export async function apiGet<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal,
    },
  )

  if (!response.ok) {
    let message =
      `Request failed with status ${response.status}`

    try {
      const body = await response.json()

      if (
        typeof body?.detail?.message === 'string'
      ) {
        message = body.detail.message
      }
    } catch {
      // Keep the HTTP fallback message.
    }

    throw new APIError(
      message,
      response.status,
    )
  }

  return response.json() as Promise<T>
}