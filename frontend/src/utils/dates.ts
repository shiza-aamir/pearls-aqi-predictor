const PAKISTAN_TIME_ZONE = 'Asia/Karachi'

export function formatDateTimePKT(
  value: string,
): string {
  return new Intl.DateTimeFormat('en-GB', {
    timeZone: PAKISTAN_TIME_ZONE,
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  }).format(new Date(value))
}

export function formatTimePKT(
  value: string,
): string {
  return new Intl.DateTimeFormat('en-GB', {
    timeZone: PAKISTAN_TIME_ZONE,
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  }).format(new Date(value))
}

export function formatShortDatePKT(
  value: string,
): string {
  return new Intl.DateTimeFormat('en-GB', {
    timeZone: PAKISTAN_TIME_ZONE,
    day: '2-digit',
    month: 'short',
  }).format(new Date(value))
}

export function addHours(
  value: string,
  hours: number,
): Date {
  const date = new Date(value)

  return new Date(
    date.getTime() + hours * 60 * 60 * 1000,
  )
}