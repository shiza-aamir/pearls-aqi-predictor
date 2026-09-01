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

interface ForecastCardsProps {
  forecasts: ForecastItem[]
}

export function ForecastCards({
  forecasts,
}: ForecastCardsProps) {
  return (
    <section
      aria-labelledby="forecast-cards-heading"
    >
      <div
        className="
          mb-4
          flex
          items-baseline
          justify-between
          gap-4
        "
      >
        <h2
          id="forecast-cards-heading"
          className="
            text-[14px]
            font-semibold
          "
        >
          Forecast detail
        </h2>

        <span
          className="
            text-[11px]
            text-[var(--color-text-tertiary)]
          "
        >
          XGBoost production models
        </span>
      </div>

      <div
        className="
          grid gap-3
          lg:grid-cols-3
        "
      >
        {forecasts.map((forecast) => {
          const presentation =
            getAQICategoryPresentation(
              forecast.category,
            )

          return (
            <article
              key={forecast.horizon_hours}
              className="
                overflow-hidden
                rounded-[6px]
                border
                border-[var(--color-border)]
                bg-[var(--color-surface)]
              "
            >
              <div
                className="h-[3px]"
                style={{
                  backgroundColor:
                    presentation.color,
                }}
              />

              <div className="p-5">
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
                      {
                        forecast.horizon_hours
                      }
                      -hour forecast
                    </p>

                    <p
                      className="
                        mt-1
                        text-[11px]
                        text-[var(--color-text-secondary)]
                      "
                    >
                      {formatDateTimePKT(
                        forecast.target_at,
                      )}
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
                    {forecast.model.alias}
                  </span>
                </div>

                <div className="mt-7">
                  <p
                    className="
                      font-display
                      text-[48px]
                      font-medium
                      leading-none
                      tracking-[-0.035em]
                    "
                  >
                    {formatAQI(
                      forecast.aqi,
                    )}
                  </p>

                  <p
                    className="
                      mt-2
                      text-[11px]
                      font-medium
                    "
                    style={{
                      color:
                        presentation.color,
                    }}
                  >
                    {presentation.label}
                  </p>
                </div>

                <div
                  className="
                    mt-6
                    border-t
                    border-[var(--color-border)]
                    pt-4
                  "
                >
                  <p
                    className="
                      text-[10px]
                      font-medium
                      text-[var(--color-text-secondary)]
                    "
                  >
                    {forecast.alert.level}
                  </p>
                </div>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}