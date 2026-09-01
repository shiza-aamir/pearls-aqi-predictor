import {
  AlertCircle,
  RefreshCw,
} from 'lucide-react'

interface DashboardErrorProps {
  message: string
  onRetry: () => void
}

export function DashboardError({
  message,
  onRetry,
}: DashboardErrorProps) {
  return (
    <div
      role="alert"
      className="
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
        p-6
      "
    >
      <div
        className="
          flex items-start gap-3
        "
      >
        <AlertCircle
          aria-hidden="true"
          size={19}
          strokeWidth={1.7}
          className="
            mt-[1px]
            shrink-0
            text-[var(--aqi-unhealthy)]
          "
        />

        <div>
          <h2
            className="
              text-[14px]
              font-semibold
            "
          >
            Air quality data is unavailable
          </h2>

          <p
            className="
              mt-1
              text-[13px]
              leading-6
              text-[var(--color-text-secondary)]
            "
          >
            {message}
          </p>

          <button
            type="button"
            onClick={onRetry}
            className="
              mt-4 inline-flex
              items-center gap-2
              rounded-[5px]
              border
              border-[var(--color-border-strong)]
              bg-[var(--color-surface)]
              px-3 py-2
              text-[12px]
              font-medium
              transition-colors
              hover:bg-[var(--color-surface-sunken)]
            "
          >
            <RefreshCw
              size={14}
              strokeWidth={1.8}
            />
            Try again
          </button>
        </div>
      </div>
    </div>
  )
}