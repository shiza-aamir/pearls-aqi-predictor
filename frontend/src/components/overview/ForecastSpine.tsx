import type {
  ForecastItem,
} from '../../api/types'
import {
  getAQICategoryPresentation,
} from '../../utils/aqi'
import {
  formatDateTimePKT,
} from '../../utils/dates'
import {
  formatAQI,
} from '../../utils/formatters'

interface ForecastSpineProps {
  currentAQI: number
  currentCategory: string
  observedAt: string
  forecasts: ForecastItem[]
}

interface SpinePoint {
  label: string
  timestamp: string
  aqi: number
  category: string
}

export function ForecastSpine({
  currentAQI,
  currentCategory,
  observedAt,
  forecasts,
}: ForecastSpineProps) {
  const points: SpinePoint[] = [
    {
      label: 'Now',
      timestamp: observedAt,
      aqi: currentAQI,
      category: currentCategory,
    },
    ...forecasts.map((forecast) => ({
      label: `+${forecast.horizon_hours}h`,
      timestamp: forecast.target_at,
      aqi: forecast.aqi,
      category: forecast.category,
    })),
  ]

  return (
    <section
      className="
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
        p-5 md:p-6
      "
      aria-labelledby="forecast-spine-heading"
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
            id="forecast-spine-heading"
            className="
              text-[14px]
              font-semibold
            "
          >
            72-hour outlook
          </h2>

          <p
            className="
              mt-1
              text-[12px]
              text-[var(--color-text-tertiary)]
            "
          >
            Forecast progression from the
            latest observation
          </p>
        </div>

        <span
          className="
            font-mono
            text-[10px]
            uppercase
            tracking-[0.06em]
            text-[var(--color-text-tertiary)]
          "
        >
          US AQI
        </span>
      </div>

      <div
        className="
          mt-7
          hidden
          md:block
        "
      >
        <div
          className="
            relative
            grid
            grid-cols-4
          "
        >
          <div
            aria-hidden="true"
            className="
              absolute
              left-[12.5%]
              right-[12.5%]
              top-[15px]
              h-px
              bg-[var(--color-border-strong)]
            "
          />

          {points.map((point) => {
            const presentation =
              getAQICategoryPresentation(
                point.category,
              )

            return (
              <div
                key={point.label}
                className="
                  relative z-10
                  flex
                  flex-col
                  items-center
                  px-2
                  text-center
                "
              >
                <div
                  className="
                    flex
                    h-[31px]
                    w-[31px]
                    items-center
                    justify-center
                    rounded-full
                    border-[5px]
                    border-[var(--color-surface)]
                  "
                  style={{
                    backgroundColor:
                      presentation.color,
                  }}
                  aria-label={`${point.label}: AQI ${formatAQI(
                    point.aqi,
                  )}, ${point.category}`}
                />

                <p
                  className="
                    mt-3
                    text-[10px]
                    font-semibold
                    uppercase
                    tracking-[0.08em]
                    text-[var(--color-text-tertiary)]
                  "
                >
                  {point.label}
                </p>

                <p
                  className="
                    font-display
                    mt-1
                    text-[29px]
                    font-medium
                    leading-none
                  "
                >
                  {formatAQI(point.aqi)}
                </p>

                <p
                  className="
                    mt-2
                    max-w-[150px]
                    text-[10px]
                    font-medium
                  "
                  style={{
                    color:
                      presentation.color,
                  }}
                >
                  {presentation.shortLabel}
                </p>

                <p
                  className="
                    mt-1.5
                    text-[9px]
                    leading-4
                    text-[var(--color-text-tertiary)]
                  "
                >
                  {formatDateTimePKT(
                    point.timestamp,
                  )}
                </p>
              </div>
            )
          })}
        </div>
      </div>

      <div
        className="
          mt-6
          space-y-0
          md:hidden
        "
      >
        {points.map((point, index) => {
          const presentation =
            getAQICategoryPresentation(
              point.category,
            )

          return (
            <div
              key={point.label}
              className="
                relative
                flex
                gap-4
                pb-6
                last:pb-0
              "
            >
              {index < points.length - 1 && (
                <div
                  aria-hidden="true"
                  className="
                    absolute
                    left-[8px]
                    top-[18px]
                    bottom-[-2px]
                    w-px
                    bg-[var(--color-border-strong)]
                  "
                />
              )}

              <div
                aria-hidden="true"
                className="
                  relative z-10
                  mt-[4px]
                  h-[17px]
                  w-[17px]
                  shrink-0
                  rounded-full
                  border-[3px]
                  border-[var(--color-surface)]
                "
                style={{
                  backgroundColor:
                    presentation.color,
                }}
              />

              <div
                className="
                  flex
                  min-w-0
                  flex-1
                  items-start
                  justify-between
                  gap-4
                "
              >
                <div>
                  <p
                    className="
                      text-[10px]
                      font-semibold
                      uppercase
                      tracking-[0.08em]
                      text-[var(--color-text-tertiary)]
                    "
                  >
                    {point.label}
                  </p>

                  <p
                    className="
                      mt-1
                      text-[11px]
                      text-[var(--color-text-secondary)]
                    "
                  >
                    {formatDateTimePKT(
                      point.timestamp,
                    )}
                  </p>
                </div>

                <div className="text-right">
                  <p
                    className="
                      font-display
                      text-[27px]
                      font-medium
                      leading-none
                    "
                  >
                    {formatAQI(point.aqi)}
                  </p>

                  <p
                    className="
                      mt-1
                      text-[10px]
                      font-medium
                    "
                    style={{
                      color:
                        presentation.color,
                    }}
                  >
                    {
                      presentation.shortLabel
                    }
                  </p>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}