import {
  Activity,
  Clock3,
} from 'lucide-react'

import type {
  LivePerformance,
  PerformanceResponse,
} from '../../api/types'
import {
  formatDecimal,
} from '../../utils/formatters'

interface LiveModelMonitoringProps {
  performance: PerformanceResponse
}

const PRELIMINARY_SAMPLE_THRESHOLD = 30

function formatMaturityTime(
  value: string | null,
): string | null {
  if (!value) {
    return null
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return null
  }

  return new Intl.DateTimeFormat(
    'en-PK',
    {
      timeZone: 'Asia/Karachi',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    },
  ).format(date)
}

function HorizonCard({
  item,
}: {
  item: LivePerformance
}) {
  const awaiting =
    item.evaluated_forecasts === 0

  const preliminary =
    item.evaluated_forecasts > 0 &&
    item.evaluated_forecasts <
      PRELIMINARY_SAMPLE_THRESHOLD

  const nextMaturity =
    formatMaturityTime(
      item.next_maturity_at,
    )

  return (
    <article
      className="
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface-sunken)]
        p-4
      "
    >
      <div
        className="
          flex
          items-start
          justify-between
          gap-3
        "
      >
        <div>
          <p
            className="
              text-[10px]
              font-semibold
              uppercase
              tracking-[0.07em]
              text-[var(--color-text-tertiary)]
            "
          >
            {item.horizon_hours}-hour
          </p>

          <p
            className="
              mt-1
              text-[10px]
              text-[var(--color-text-secondary)]
            "
          >
            {awaiting
              ? 'Awaiting first evaluation'
              : 'Live evaluation available'}
          </p>
        </div>

        {awaiting ? (
          <Clock3
            size={15}
            strokeWidth={1.8}
            className="
              text-[var(--color-text-tertiary)]
            "
          />
        ) : (
          <Activity
            size={15}
            strokeWidth={1.8}
            className="
              text-[var(--color-accent-strong)]
            "
          />
        )}
      </div>

      {awaiting ? (
        <div className="mt-5">
          <p
            className="
              text-[11px]
              text-[var(--color-text-secondary)]
            "
          >
            No matured forecasts yet
          </p>
        </div>
      ) : (
        <>
          <div
            className="
              mt-5
              grid
              grid-cols-2
              gap-4
            "
          >
            <div>
              <p
                className="
                  text-[9px]
                  font-semibold
                  uppercase
                  tracking-[0.07em]
                  text-[var(--color-text-tertiary)]
                "
              >
                MAE
              </p>

              <p
                className="
                  font-mono
                  mt-1
                  text-[16px]
                  font-medium
                "
              >
                {item.mae === null
                  ? '—'
                  : formatDecimal(
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
                  tracking-[0.07em]
                  text-[var(--color-text-tertiary)]
                "
              >
                RMSE
              </p>

              <p
                className="
                  font-mono
                  mt-1
                  text-[16px]
                  font-medium
                "
              >
                {item.rmse === null
                  ? '—'
                  : formatDecimal(
                      item.rmse,
                      2,
                    )}
              </p>
            </div>
          </div>

          <p
            className="
              mt-4
              text-[10px]
              text-[var(--color-text-tertiary)]
            "
          >
            {item.evaluated_forecasts}{' '}
            matured forecast
            {item.evaluated_forecasts === 1
              ? ''
              : 's'}
          </p>

          {preliminary && (
            <p
              className="
                mt-1
                text-[9px]
                text-[var(--color-text-tertiary)]
              "
            >
              Preliminary sample
            </p>
          )}
        </>
      )}

      <div
        className="
          mt-4
          border-t
          border-[var(--color-border)]
          pt-3
        "
      >
        <p
          className="
            text-[9px]
            font-semibold
            uppercase
            tracking-[0.07em]
            text-[var(--color-text-tertiary)]
          "
        >
          Next pending forecast
        </p>

        <p
          className="
            mt-1
            text-[11px]
            font-medium
            text-[var(--color-text-secondary)]
          "
        >
          {nextMaturity
            ? `${nextMaturity} PKT`
            : 'No pending maturity'}
        </p>
      </div>
    </article>
  )
}

export function LiveModelMonitoring({
  performance,
}: LiveModelMonitoringProps) {
  const evaluated =
    performance.live_evaluated_forecasts

  const anyLive =
    performance.live.some(
      (item) =>
        item.evaluated_forecasts > 0,
    )

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
            {anyLive ? (
              <Activity
                size={17}
                strokeWidth={1.8}
              />
            ) : (
              <Clock3
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
              Each horizon is evaluated
              independently when its target
              time matures and the matching
              observed AQI becomes available.
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
            <HorizonCard
              key={item.horizon_hours}
              item={item}
            />
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
          Live production metrics are
          preliminary while the number of
          matured forecasts is small. The
          frozen final holdout remains the
          stable model-evaluation reference.
        </p>
      )}
    </section>
  )
}