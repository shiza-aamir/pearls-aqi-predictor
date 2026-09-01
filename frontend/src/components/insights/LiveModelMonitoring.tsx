import {
  Activity,
  Clock3,
} from 'lucide-react'

import type {
  PerformanceResponse,
} from '../../api/types'
import {
  formatDecimal,
} from '../../utils/formatters'

interface LiveModelMonitoringProps {
  performance: PerformanceResponse
}

const PRELIMINARY_SAMPLE_THRESHOLD = 30

export function LiveModelMonitoring({
  performance,
}: LiveModelMonitoringProps) {
  const evaluated =
    performance.live_evaluated_forecasts

  const awaiting =
    performance.live_status ===
      'awaiting_matured_forecasts' ||
    evaluated === 0

  const preliminary =
    evaluated > 0 &&
    evaluated <
      PRELIMINARY_SAMPLE_THRESHOLD

  return (
    <section
      className="
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
        p-5
        md:p-6
      "
      aria-labelledby="live-monitoring-heading"
    >
      <div
        className="
          flex
          flex-col
          gap-5
          lg:flex-row
          lg:items-center
          lg:justify-between
        "
      >
        <div
          className="
            flex
            items-start
            gap-3
          "
        >
          <div
            className="
              flex
              h-9
              w-9
              shrink-0
              items-center
              justify-center
              rounded-[5px]
              bg-[var(--color-accent-soft)]
              text-[var(--color-accent-strong)]
            "
          >
            {awaiting ? (
              <Clock3
                size={17}
                strokeWidth={1.8}
              />
            ) : (
              <Activity
                size={17}
                strokeWidth={1.8}
              />
            )}
          </div>

          <div>
            <div
              className="
                flex
                flex-wrap
                items-center
                gap-2
              "
            >
              <h2
                id="live-monitoring-heading"
                className="
                  text-[14px]
                  font-semibold
                "
              >
                Live model monitoring
              </h2>

              {preliminary && (
                <span
                  className="
                    rounded-[4px]
                    bg-[var(--color-surface-sunken)]
                    px-2
                    py-1
                    text-[8px]
                    font-semibold
                    uppercase
                    tracking-[0.07em]
                    text-[var(--color-text-tertiary)]
                  "
                >
                  Preliminary
                </span>
              )}
            </div>

            <p
              className="
                mt-1
                max-w-3xl
                text-[11px]
                leading-5
                text-[var(--color-text-secondary)]
              "
            >
              {awaiting
                ? 'Production forecasts are being recorded. Metrics will appear after forecast target times mature and matching actual AQI observations are available.'
                : 'Metrics below are calculated only from matured production forecasts with observed outcomes.'}
            </p>
          </div>
        </div>

        <div
          className="
            shrink-0
            lg:text-right
          "
        >
          <p
            className="
              font-display
              text-[34px]
              font-medium
              leading-none
            "
          >
            {evaluated}
          </p>

          <p
            className="
              mt-1
              text-[9px]
              uppercase
              tracking-[0.07em]
              text-[var(--color-text-tertiary)]
            "
          >
            matured forecasts
          </p>
        </div>
      </div>

      {!awaiting && (
        <>
          <div
            className="
              mt-6
              grid
              gap-3
              border-t
              border-[var(--color-border)]
              pt-5
              lg:grid-cols-3
            "
          >
            {performance.live.map(
              (item) => (
                <div
                  key={item.horizon_hours}
                  className="
                    rounded-[5px]
                    bg-[var(--color-surface-sunken)]
                    p-4
                  "
                >
                  <p
                    className="
                      text-[10px]
                      font-semibold
                      uppercase
                      tracking-[0.07em]
                      text-[var(--color-text-tertiary)]
                    "
                  >
                    {item.horizon_hours}
                    -hour
                  </p>

                  <p
                    className="
                      font-mono
                      mt-3
                      text-[14px]
                      font-medium
                    "
                  >
                    {item.mae === null
                      ? 'MAE —'
                      : `MAE ${formatDecimal(
                          item.mae,
                          2,
                        )}`}
                  </p>

                  <p
                    className="
                      mt-1
                      text-[10px]
                      text-[var(--color-text-tertiary)]
                    "
                  >
                    {
                      item.evaluated_forecasts
                    }{' '}
                    evaluated
                  </p>
                </div>
              ),
            )}
          </div>

          {preliminary && (
            <p
              className="
                mt-4
                text-[9px]
                leading-5
                text-[var(--color-text-tertiary)]
              "
            >
              Live metrics are preliminary
              while the number of matured
              forecasts is small. They should
              not yet be treated as a stable
              estimate of production
              performance.
            </p>
          )}
        </>
      )}
    </section>
  )
}