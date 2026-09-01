export function formatAQI(
  value: number,
): string {
  return Math.round(value).toString()
}

export function formatDecimal(
  value: number,
  digits = 1,
): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function formatPercentage(
  value: number,
  digits = 1,
): string {
  return `${formatDecimal(value, digits)}%`
}

export function formatInteger(
  value: number,
): string {
  return Math.round(value).toLocaleString('en-US')
}

export function formatSignedNumber(
  value: number,
  digits = 2,
): string {
  const formatted = Math.abs(value).toFixed(digits)

  if (value > 0) {
    return `+${formatted}`
  }

  if (value < 0) {
    return `-${formatted}`
  }

  return formatted
}