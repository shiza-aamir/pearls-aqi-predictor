import type {
  HistoryStatistics as HistoryStatisticsType,
} from '../../api/types'
import {
  formatDecimal,
  formatAQI,
} from '../../utils/formatters'

interface HistoryStatisticsProps {
  statistics: HistoryStatisticsType
}

interface StatisticProps {
  label: string
  value: string
  description: string
}

function Statistic({
  label,
  value,
  description,
}: StatisticProps) {
  return (
    <div
      className="
        min-w-0
        border-[var(--color-border)]
        lg:border-r
        lg:pr-5
        lg:last:border-r-0
        lg:last:pr-0
      "
    >
      <p
        className="
          text-[10px]
          font-semibold
          uppercase
          tracking-[0.08em]
          text-[var(--color-text-tertiary)]
        "
      >
        {label}
      </p>

      <p
        className="
          font-display
          mt-2
          text-[32px]
          font-medium
          leading-none
          text-[var(--color-text-primary)]
        "
      >
        {value}
      </p>

      <p
        className="
          mt-2
          text-[10px]
          text-[var(--color-text-tertiary)]
        "
      >
        {description}
      </p>
    </div>
  )
}

export function HistoryStatistics({
  statistics,
}: HistoryStatisticsProps) {
  return (
    <section
      className="
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
        p-5 md:p-6
      "
      aria-labelledby="history-statistics-heading"
    >
      <div className="mb-6">
        <h2
          id="history-statistics-heading"
          className="
            text-[14px]
            font-semibold
          "
        >
          Period summary
        </h2>

        <p
          className="
            mt-1
            text-[11px]
            text-[var(--color-text-tertiary)]
          "
        >
          AQI statistics for the selected
          observation window
        </p>
      </div>

      <div
        className="
          grid
          grid-cols-2
          gap-x-6
          gap-y-7
          lg:grid-cols-4
          lg:gap-y-0
        "
      >
        <Statistic
          label="Minimum"
          value={formatAQI(
            statistics.minimum,
          )}
          description="Lowest AQI"
        />

        <Statistic
          label="Average"
          value={formatDecimal(
            statistics.average,
            1,
          )}
          description="Mean AQI"
        />

        <Statistic
          label="Maximum"
          value={formatAQI(
            statistics.maximum,
          )}
          description="Highest AQI"
        />

        <Statistic
          label="Variability"
          value={formatDecimal(
            statistics.standard_deviation,
            1,
          )}
          description="Standard deviation"
        />
      </div>
    </section>
  )
}