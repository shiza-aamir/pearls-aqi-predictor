import type {
  HistoryHours,
} from '../../api/history'

interface HistoryRangeSelectorProps {
  value: HistoryHours
  onChange: (hours: HistoryHours) => void
}

const OPTIONS: Array<{
  label: string
  value: HistoryHours
}> = [
  {
    label: '24h',
    value: 24,
  },
  {
    label: '48h',
    value: 48,
  },
  {
    label: '72h',
    value: 72,
  },
  {
    label: '7 days',
    value: 168,
  },
]

export function HistoryRangeSelector({
  value,
  onChange,
}: HistoryRangeSelectorProps) {
  return (
    <div
      role="group"
      aria-label="History range"
      className="
        inline-flex
        rounded-[6px]
        border
        border-[var(--color-border-strong)]
        bg-[var(--color-surface)]
        p-1
      "
    >
      {OPTIONS.map((option) => {
        const active =
          option.value === value

        return (
          <button
            key={option.value}
            type="button"
            onClick={() =>
              onChange(option.value)
            }
            className={`
              rounded-[4px]
              px-3
              py-2
              text-[11px]
              font-medium
              transition-colors

              ${
                active
                  ? `
                    bg-[var(--color-accent-soft)]
                    text-[var(--color-accent-strong)]
                  `
                  : `
                    text-[var(--color-text-secondary)]
                    hover:bg-[var(--color-surface-sunken)]
                  `
              }
            `}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}