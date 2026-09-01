import {
  CheckCircle2,
  Info,
} from 'lucide-react'

interface HistoryAvailabilityProps {
  requestedHours: number
  availableHours: number
}

export function HistoryAvailability({
  requestedHours,
  availableHours,
}: HistoryAvailabilityProps) {
  const complete =
    availableHours >= requestedHours

  return (
    <div
      className="
        flex
        items-start
        gap-3
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
        px-4
        py-3
      "
    >
      <div
        className="
          mt-[1px]
          shrink-0
          text-[var(--color-accent-strong)]
        "
      >
        {complete ? (
          <CheckCircle2
            size={16}
            strokeWidth={1.8}
          />
        ) : (
          <Info
            size={16}
            strokeWidth={1.8}
          />
        )}
      </div>

      <div>
        <p
          className="
            text-[11px]
            font-medium
            text-[var(--color-text-primary)]
          "
        >
          {availableHours} of{' '}
          {requestedHours} requested hours
          available
        </p>

        {!complete && (
          <p
            className="
              mt-1
              max-w-3xl
              text-[10px]
              leading-5
              text-[var(--color-text-tertiary)]
            "
          >
            The chart shows only AQI-ready
            observations currently available
            from the stored history. No
            missing observations are
            fabricated or extrapolated.
          </p>
        )}
      </div>
    </div>
  )
}