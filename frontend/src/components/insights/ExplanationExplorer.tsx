import {
  useMemo,
  useState,
} from 'react'
import {
  ArrowDown,
  ArrowUp,
  BrainCircuit,
} from 'lucide-react'

import type {
  ForecastItem,
  ForecastResponse,
} from '../../api/types'
import {
  getAQICategoryPresentation,
} from '../../utils/aqi'
import {
  formatDateTimePKT,
} from '../../utils/dates'
import {
  formatAQI,
  formatDecimal,
  formatSignedNumber,
} from '../../utils/formatters'

interface ExplanationExplorerProps {
  forecast: ForecastResponse
}

type Horizon = 24 | 48 | 72

const HORIZONS: Horizon[] = [
  24,
  48,
  72,
]

function getForecastForHorizon(
  forecasts: ForecastItem[],
  horizon: Horizon,
): ForecastItem | undefined {
  return forecasts.find(
    (item) =>
      item.horizon_hours === horizon,
  )
}

export function ExplanationExplorer({
  forecast,
}: ExplanationExplorerProps) {
  const [selectedHorizon, setSelectedHorizon] =
    useState<Horizon>(24)

  const selectedForecast = useMemo(
    () =>
      getForecastForHorizon(
        forecast.forecasts,
        selectedHorizon,
      ),
    [
      forecast.forecasts,
      selectedHorizon,
    ],
  )

  if (!selectedForecast) {
    return null
  }

  const contributions =
    selectedForecast.explanation.top_features

  const maximumContribution = Math.max(
    ...contributions.map((item) =>
      Math.abs(item.contribution),
    ),
    1,
  )

  const presentation =
    getAQICategoryPresentation(
      selectedForecast.category,
    )

  return (
    <section
      className="
        overflow-hidden
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
      "
      aria-labelledby="forecast-explanation-heading"
    >
      <div
        className="
          border-b
          border-[var(--color-border)]
          p-5
          md:p-6
        "
      >
        <div
          className="
            flex
            flex-col
            gap-4
            lg:flex-row
            lg:items-start
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
              <BrainCircuit
                size={17}
                strokeWidth={1.7}
              />
            </div>

            <div>
              <h2
                id="forecast-explanation-heading"
                className="
                  text-[14px]
                  font-semibold
                "
              >
                Forecast explanation
              </h2>

              <p
                className="
                  mt-1
                  max-w-2xl
                  text-[11px]
                  leading-5
                  text-[var(--color-text-tertiary)]
                "
              >
                Inspect the strongest feature
                contributions shaping each
                production forecast for{' '}
                {forecast.city}.
              </p>
            </div>
          </div>

          <div
            role="group"
            aria-label="Forecast explanation horizon"
            className="
              inline-flex
              self-start
              rounded-[6px]
              border
              border-[var(--color-border-strong)]
              bg-[var(--color-surface)]
              p-1
            "
          >
            {HORIZONS.map((horizon) => {
              const active =
                selectedHorizon === horizon

              return (
                <button
                  key={horizon}
                  type="button"
                  onClick={() =>
                    setSelectedHorizon(
                      horizon,
                    )
                  }
                  className={`
                    rounded-[4px]
                    px-4
                    py-2
                    text-[10px]
                    font-semibold
                    uppercase
                    tracking-[0.06em]
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
                  {horizon}h
                </button>
              )
            })}
          </div>
        </div>
      </div>

      <div
        className="
          grid
          lg:grid-cols-[0.78fr_1.6fr]
        "
      >
        <div
          className="
            border-b
            border-[var(--color-border)]
            p-5
            md:p-6
            lg:border-b-0
            lg:border-r
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
            {selectedHorizon}-hour forecast
          </p>

          <div
            className="
              mt-5
              flex
              items-end
              gap-3
            "
          >
            <p
              className="
                font-display
                text-[56px]
                font-medium
                leading-none
                tracking-[-0.04em]
              "
            >
              {formatAQI(
                selectedForecast.aqi,
              )}
            </p>

            <span
              className="
                mb-1
                text-[10px]
                uppercase
                tracking-[0.06em]
                text-[var(--color-text-tertiary)]
              "
            >
              US AQI
            </span>
          </div>

          <p
            className="
              mt-3
              text-[11px]
              font-medium
            "
            style={{
              color: presentation.color,
            }}
          >
            {presentation.label}
          </p>

          <div
            className="
              mt-7
              space-y-5
              border-t
              border-[var(--color-border)]
              pt-5
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
                Forecast target
              </p>

              <p
                className="
                  mt-1
                  text-[11px]
                  leading-5
                  text-[var(--color-text-secondary)]
                "
              >
                {formatDateTimePKT(
                  selectedForecast.target_at,
                )}{' '}
                PKT
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
                Model baseline
              </p>

              <p
                className="
                  font-mono
                  mt-1
                  text-[16px]
                  font-medium
                "
              >
                {formatDecimal(
                  selectedForecast
                    .explanation
                    .base_value,
                  1,
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
                Production alias
              </p>

              <span
                className="
                  font-mono
                  mt-1
                  inline-block
                  rounded-[4px]
                  bg-[var(--color-surface-sunken)]
                  px-2
                  py-1
                  text-[9px]
                  text-[var(--color-text-secondary)]
                "
              >
                {
                  selectedForecast
                    .model.alias
                }
              </span>
            </div>
          </div>
        </div>

        <div className="p-5 md:p-6">
          <div
            className="
              mb-6
              flex
              items-baseline
              justify-between
              gap-4
            "
          >
            <div>
              <h3
                className="
                  text-[12px]
                  font-semibold
                "
              >
                Top feature contributions
              </h3>

              <p
                className="
                  mt-1
                  text-[10px]
                  text-[var(--color-text-tertiary)]
                "
              >
                Top SHAP features for this
                prediction
              </p>
            </div>

            <span
              className="
                font-mono
                text-[9px]
                text-[var(--color-text-tertiary)]
              "
            >
              SHAP
            </span>
          </div>

          <div className="space-y-5">
            {contributions.map((item) => {
              const increasing =
                item.direction === 'increase'

              const magnitude =
                Math.abs(
                  item.contribution,
                )

              const width =
                (magnitude /
                  maximumContribution) *
                100

              return (
                <div key={item.feature}>
                  <div
                    className="
                      flex
                      items-start
                      justify-between
                      gap-5
                    "
                  >
                    <div className="min-w-0">
                      <p
                        className="
                          text-[11px]
                          font-medium
                          text-[var(--color-text-primary)]
                        "
                      >
                        {item.display_name}
                      </p>

                      <p
                        className="
                          font-mono
                          mt-1
                          text-[9px]
                          text-[var(--color-text-tertiary)]
                        "
                      >
                        feature value{' '}
                        {formatDecimal(
                          item.feature_value,
                          2,
                        )}
                      </p>
                    </div>

                    <div
                      className="
                        flex
                        shrink-0
                        items-center
                        gap-1.5
                      "
                      style={{
                        color: increasing
                          ? 'var(--aqi-unhealthy)'
                          : 'var(--color-accent-strong)',
                      }}
                    >
                      {increasing ? (
                        <ArrowUp
                          size={12}
                          strokeWidth={1.8}
                        />
                      ) : (
                        <ArrowDown
                          size={12}
                          strokeWidth={1.8}
                        />
                      )}

                      <span
                        className="
                          font-mono
                          text-[11px]
                          font-medium
                        "
                      >
                        {formatSignedNumber(
                          item.contribution,
                          2,
                        )}
                      </span>
                    </div>
                  </div>

                  <div
                    className="
                      mt-2
                      h-[5px]
                      overflow-hidden
                      rounded-full
                      bg-[var(--color-surface-sunken)]
                    "
                  >
                    <div
                      className="
                        h-full
                        rounded-full
                      "
                      style={{
                        width: `${width}%`,
                        backgroundColor:
                          increasing
                            ? 'var(--aqi-unhealthy)'
                            : 'var(--color-accent-strong)',
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div
        className="
          border-t
          border-[var(--color-border)]
          bg-[var(--color-surface-sunken)]
          px-5
          py-4
          md:px-6
        "
      >
        <div
          className="
            flex
            flex-col
            gap-2
            sm:flex-row
            sm:items-center
            sm:justify-between
          "
        >
          <div
            className="
              flex
              flex-wrap
              gap-x-5
              gap-y-2
              text-[9px]
              text-[var(--color-text-tertiary)]
            "
          >
            <span
              className="
                flex
                items-center
                gap-1.5
              "
            >
              <ArrowUp
                size={11}
                strokeWidth={1.8}
                className="
                  text-[var(--aqi-unhealthy)]
                "
              />
              Pushes forecast higher
            </span>

            <span
              className="
                flex
                items-center
                gap-1.5
              "
            >
              <ArrowDown
                size={11}
                strokeWidth={1.8}
                className="
                  text-[var(--color-accent-strong)]
                "
              />
              Pushes forecast lower
            </span>
          </div>

          <p
            className="
              text-[9px]
              leading-5
              text-[var(--color-text-tertiary)]
            "
          >
            SHAP explains model behaviour,
            not causal effects.
          </p>
        </div>
      </div>
    </section>
  )
}