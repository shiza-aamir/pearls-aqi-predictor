import {
  Activity,
  CheckCircle2,
  Clock3,
} from 'lucide-react'

import type {
  PerformanceResponse,
} from '../../api/types'
import {
  formatDecimal,
} from '../../utils/formatters'

interface ForecastReliabilityProps {
  performance: PerformanceResponse
}

function formatPercent(
  value: number,
): string {
  return `${value.toFixed(1)}%`
}

export function ForecastReliability({
  performance,
}: ForecastReliabilityProps) {
  const awaitingLiveData =
    performance.live_status ===
      'awaiting_matured_forecasts' ||
    performance.live_evaluated_forecasts === 0

  const preliminaryLiveData =
    !awaitingLiveData &&
    performance.live_evaluated_forecasts < 30

  return (
    <section
      aria-labelledby="forecast-reliability-heading"
      className="space-y-4"
    >
      <div
        className="
          flex flex-col gap-2
          sm:flex-row
          sm:items-end
          sm:justify-between
        "
      >
        <div>
          <h2
            id="forecast-reliability-heading"
            className="
              text-[14px]
              font-semibold
              text-[var(--color-text-primary)]
            "
          >
            Forecast reliability
          </h2>

          <p
            className="
              mt-1
              text-[12px]
              text-[var(--color-text-tertiary)]
            "
          >
            Performance measured on the
            frozen final holdout set
          </p>
        </div>

        <div
          className="
            inline-flex
            items-center gap-2
            text-[10px]
            uppercase
            tracking-[0.07em]
            text-[var(--color-text-tertiary)]
          "
        >
          <CheckCircle2
            size={13}
            strokeWidth={1.8}
          />

          {performance.holdout_evaluation_label}
        </div>
      </div>

      <div
        className="
          grid gap-3
          lg:grid-cols-3
        "
      >
        {performance.holdout.map((item) => (
          <article
            key={item.horizon_hours}
            className="
              rounded-[6px]
              border
              border-[var(--color-border)]
              bg-[var(--color-surface)]
              p-5
            "
          >
            <div
              className="
                flex
                items-start
                justify-between
                gap-4
              "
            >
              <div>
                <p
                  className="
                    text-[11px]
                    font-semibold
                    uppercase
                    tracking-[0.08em]
                    text-[var(--color-text-tertiary)]
                  "
                >
                  {item.horizon_hours}-hour
                </p>

                <p
                  className="
                    mt-1
                    text-[11px]
                    text-[var(--color-text-secondary)]
                  "
                >
                  Final holdout evaluation
                </p>
              </div>

              <span
                className="
                  font-mono
                  rounded-[4px]
                  bg-[var(--color-surface-sunken)]
                  px-2 py-1
                  text-[9px]
                  text-[var(--color-text-tertiary)]
                "
              >
                {item.rows.toLocaleString()}
                {' '}rows
              </span>
            </div>

            <div className="mt-7">
              <p
                className="
                  font-display
                  text-[42px]
                  font-medium
                  leading-none
                  tracking-[-0.035em]
                  text-[var(--color-text-primary)]
                "
              >
                {formatPercent(
                  item.within_30_aqi_pct,
                )}
              </p>

              <p
                className="
                  mt-2
                  text-[12px]
                  font-medium
                  text-[var(--color-accent-strong)]
                "
              >
                within ±30 AQI points
              </p>
            </div>

            <div
              className="
                mt-6 grid
                grid-cols-2
                gap-4
                border-t
                border-[var(--color-border)]
                pt-4
              "
            >
              <div>
                <p
                  className="
                    text-[9px]
                    font-semibold
                    uppercase
                    tracking-[0.08em]
                    text-[var(--color-text-tertiary)]
                  "
                >
                  MAE
                </p>

                <p
                  className="
                    font-mono
                    mt-1
                    text-[15px]
                    font-medium
                  "
                >
                  {formatDecimal(
                    item.mae,
                    2,
                  )}
                </p>
              </div>

              <div>
                <p
                  className="
                    text-[9px]
                    font-semibold
                    uppercase
                    tracking-[0.08em]
                    text-[var(--color-text-tertiary)]
                  "
                >
                  Health band match
                </p>

                <p
                  className="
                    font-mono
                    mt-1
                    text-[15px]
                    font-medium
                  "
                >
                  {formatPercent(
                    item.category_accuracy_pct,
                  )}
                </p>
              </div>
            </div>

            <p
              className="
                mt-4
                text-[10px]
                leading-5
                text-[var(--color-text-tertiary)]
              "
            >
              MAE improved{' '}
              {formatPercent(
                item.mae_improvement_percent,
              )}{' '}
              over the persistence baseline.
            </p>
          </article>
        ))}
      </div>

      <div
        className="
          rounded-[6px]
          border
          border-[var(--color-border)]
          bg-[var(--color-surface)]
          p-5
          md:p-6
        "
      >
        <div
          className="
            flex flex-col gap-5
            md:flex-row
            md:items-start
            md:justify-between
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
                flex h-9 w-9
                shrink-0
                items-center
                justify-center
                rounded-[5px]
                bg-[var(--color-accent-soft)]
                text-[var(--color-accent-strong)]
              "
            >
              {awaitingLiveData ? (
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
                <h3
                  className="
                    text-[13px]
                    font-semibold
                    text-[var(--color-text-primary)]
                  "
                >
                  Live forecast monitoring
                </h3>

                {preliminaryLiveData && (
                  <span
                    className="
                      rounded-[4px]
                      border
                      border-[var(--color-border)]
                      bg-[var(--color-surface-sunken)]
                      px-2
                      py-1
                      text-[8px]
                      font-semibold
                      uppercase
                      tracking-[0.08em]
                      text-[var(--color-text-tertiary)]
                    "
                  >
                    Preliminary
                  </span>
                )}
              </div>

              {awaitingLiveData ? (
                <p
                  className="
                    mt-1
                    max-w-2xl
                    text-[12px]
                    leading-5
                    text-[var(--color-text-secondary)]
                  "
                >
                  Production forecasts are
                  being recorded. Live
                  performance will appear
                  after forecast target times
                  are reached and actual AQI
                  observations become
                  available.
                </p>
              ) : preliminaryLiveData ? (
                <p
                  className="
                    mt-1
                    max-w-2xl
                    text-[12px]
                    leading-5
                    text-[var(--color-text-secondary)]
                  "
                >
                  Live metrics are calculated
                  from matured production
                  forecasts with observed
                  outcomes. The current sample
                  is still too small for stable
                  performance conclusions.
                </p>
              ) : (
                <p
                  className="
                    mt-1
                    max-w-2xl
                    text-[12px]
                    leading-5
                    text-[var(--color-text-secondary)]
                  "
                >
                  Live metrics are calculated
                  only from matured production
                  forecasts with observed
                  outcomes.
                </p>
              )}
            </div>
          </div>

          <div
            className="
              shrink-0
              md:text-right
            "
          >
            <p
              className="
                font-display
                text-[32px]
                font-medium
                leading-none
                text-[var(--color-text-primary)]
              "
            >
              {
                performance
                  .live_evaluated_forecasts
              }
            </p>

            <p
              className="
                mt-1
                text-[10px]
                uppercase
                tracking-[0.07em]
                text-[var(--color-text-tertiary)]
              "
            >
              matured forecasts
            </p>
          </div>
        </div>

        {!awaitingLiveData && (
          <div
            className="
              mt-6 grid
              gap-3
              border-t
              border-[var(--color-border)]
              pt-5
              sm:grid-cols-3
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
                    {item.horizon_hours} hours
                  </p>

                  <p
                    className="
                      font-mono
                      mt-2
                      text-[14px]
                      font-medium
                    "
                  >
                    {item.mae === null
                      ? '—'
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
        )}
      </div>

      <p
        className="
          text-[10px]
          leading-5
          text-[var(--color-text-tertiary)]
        "
      >
        “Within ±30 AQI points” shows the
        share of forecasts whose numerical
        prediction was within 30 AQI points
        of the observed value. Health band
        match shows how often the forecast
        and observed AQI fell in the same
        AQI health-risk band.
      </p>
    </section>
  )
}