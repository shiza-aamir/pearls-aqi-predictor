import type {
  HoldoutPerformance,
} from '../../api/types'
import {
  formatDecimal,
} from '../../utils/formatters'

interface HoldoutEvaluationTableProps {
  rows: HoldoutPerformance[]
}

export function HoldoutEvaluationTable({
  rows,
}: HoldoutEvaluationTableProps) {
  return (
    <section
      className="
        overflow-hidden
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
      "
      aria-labelledby="holdout-evaluation-heading"
    >
      <div className="p-5 md:p-6">
        <h2
          id="holdout-evaluation-heading"
          className="
            text-[14px]
            font-semibold
          "
        >
          Final holdout evaluation
        </h2>

        <p
          className="
            mt-1
            text-[11px]
            text-[var(--color-text-tertiary)]
          "
        >
          One-time final test results for the
          selected production models
        </p>
      </div>

      <div className="overflow-x-auto">
        <table
          className="
            w-full
            min-w-[920px]
            border-collapse
            text-left
          "
        >
          <thead>
            <tr
              className="
                border-y
                border-[var(--color-border)]
                bg-[var(--color-surface-sunken)]
              "
            >
              {[
                'Horizon',
                'MAE',
                'RMSE',
                'R²',
                'Within ±30',
                'AQI health band match',
                'Baseline MAE',
                'Improvement',
              ].map((heading) => (
                <th
                  key={heading}
                  className="
                    px-5
                    py-3
                    text-[9px]
                    font-semibold
                    uppercase
                    tracking-[0.07em]
                    text-[var(--color-text-tertiary)]
                  "
                >
                  {heading}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((item) => (
              <tr
                key={item.horizon_hours}
                className="
                  border-b
                  border-[var(--color-border)]
                  last:border-b-0
                "
              >
                <td
                  className="
                    px-5
                    py-4
                    text-[11px]
                    font-semibold
                  "
                >
                  {item.horizon_hours}h
                </td>

                <td className="font-mono px-5 py-4 text-[11px]">
                  {formatDecimal(item.mae, 2)}
                </td>

                <td className="font-mono px-5 py-4 text-[11px]">
                  {formatDecimal(item.rmse, 2)}
                </td>

                <td className="font-mono px-5 py-4 text-[11px]">
                  {formatDecimal(item.r2, 3)}
                </td>

                <td className="font-mono px-5 py-4 text-[11px]">
                  {item.within_30_aqi_pct.toFixed(1)}%
                </td>

                <td className="font-mono px-5 py-4 text-[11px]">
                  {item.category_accuracy_pct.toFixed(1)}%
                </td>

                <td className="font-mono px-5 py-4 text-[11px]">
                  {formatDecimal(
                    item.persistence_mae,
                    2,
                  )}
                </td>

                <td
                  className="
                    font-mono
                    px-5
                    py-4
                    text-[11px]
                    text-[var(--color-accent-strong)]
                  "
                >
                  +
                  {item.mae_improvement_percent.toFixed(
                    1,
                  )}
                  %
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div
        className="
          border-t
          border-[var(--color-border)]
          px-5
          py-4
          md:px-6
        "
      >
        <p
          className="
            text-[10px]
            leading-5
            text-[var(--color-text-secondary)]
          "
        >
          <strong className="font-semibold text-[var(--color-text-primary)]">
            Within ±30
          </strong>{' '}
          shows the share of forecasts that were within
          30 AQI points of the observed value.
          {' '}
          <strong className="font-semibold text-[var(--color-text-primary)]">
            AQI health band match
          </strong>{' '}
          shows how often the forecast and the observed AQI
          fell in the same health-risk band, such as Good,
          Moderate, Unhealthy for Sensitive Groups, or
          Unhealthy.
        </p>

        <p
          className="
            mt-2
            text-[9px]
            leading-5
            text-[var(--color-text-tertiary)]
          "
        >
          A forecast can be numerically close to the observed
          AQI but still miss the health band when the value is
          near a category boundary.
        </p>
      </div>
    </section>
  )
}