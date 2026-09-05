import { RefreshCw } from 'lucide-react'

import { CitySelector } from '../components/overview/CitySelector'
import { CurrentAQICard } from '../components/overview/CurrentAQICard'
import { CurrentConditions } from '../components/overview/CurrentConditions'
import { ForecastCards } from '../components/overview/ForecastCards'
import { ForecastExplanation } from '../components/overview/ForecastExplanation'
import { ForecastReliability } from '../components/overview/ForecastReliability'
import { ForecastSpine } from '../components/overview/ForecastSpine'
import { DashboardError } from '../components/shared/DashboardError'
import { DashboardLoading } from '../components/shared/DashboardLoading'
import { useOverviewData } from '../hooks/useOverviewData'
import { formatDateTimePKT } from '../utils/dates'

export function OverviewPage() {
  const {
    cities,
    selectedCity,

    forecast,
    performance,

    citiesLoading,
    forecastLoading,
    performanceLoading,

    forecastError,
    performanceError,

    selectCity,
    refresh,
  } = useOverviewData()

  const forecast24 =
    forecast?.forecasts.find(
      (item) => item.horizon_hours === 24,
    )

  return (
    <div
      className="
        page-container
        py-8
        md:py-11
      "
    >
      <div
        className="
          mb-8
          flex
          flex-col
          gap-5
          sm:flex-row
          sm:items-end
          sm:justify-between
        "
      >
        <div>
          <p
            className="
              mb-2
              text-[11px]
              font-semibold
              uppercase
              tracking-[0.12em]
              text-[var(--color-accent)]
            "
          >
            Live air quality
          </p>

          <h1
            className="
              font-display
              text-[34px]
              font-medium
              leading-tight
              tracking-[-0.025em]
              md:text-[42px]
            "
          >
            {selectedCity}
          </h1>

          {forecast && (
            <p
              className="
                mt-2
                text-[11px]
                text-[var(--color-text-tertiary)]
              "
            >
              Latest observation ·{' '}
              {formatDateTimePKT(
                forecast.observed_at,
              )}{' '}
              PKT
            </p>
          )}
        </div>

        <div
          className="
            flex
            items-center
            gap-2
          "
        >
          <button
            type="button"
            onClick={refresh}
            disabled={forecastLoading}
            aria-label="Refresh forecast"
            title="Refresh forecast"
            className="
              flex
              h-[38px]
              w-[38px]
              items-center
              justify-center
              rounded-[5px]
              border
              border-[var(--color-border-strong)]
              bg-[var(--color-surface)]
              text-[var(--color-text-secondary)]
              transition-colors
              hover:bg-[var(--color-surface-sunken)]
              disabled:cursor-not-allowed
              disabled:opacity-50
            "
          >
            <RefreshCw
              size={15}
              strokeWidth={1.8}
              className={
                forecastLoading
                  ? 'animate-spin'
                  : ''
              }
            />
          </button>

          <CitySelector
            cities={cities}
            selectedCity={selectedCity}
            loading={citiesLoading}
            onChange={selectCity}
          />
        </div>
      </div>

      {forecastError && !forecast ? (
        <DashboardError
          message={forecastError}
          onRetry={refresh}
        />
      ) : forecastLoading && !forecast ? (
        <DashboardLoading />
      ) : forecast ? (
        <div className="space-y-6">
          {forecastError && (
            <div
              role="status"
              className="
                rounded-[5px]
                border
                border-[var(--color-border)]
                bg-[var(--color-surface-sunken)]
                px-4
                py-3
                text-[12px]
                text-[var(--color-text-secondary)]
              "
            >
              Refresh failed. Showing the
              most recent available forecast.
            </div>
          )}

          <CurrentAQICard
            city={forecast.city}
            current={forecast.current}
          />

          <ForecastSpine
            currentAQI={
              forecast.current.aqi
            }
            currentCategory={
              forecast.current.category
            }
            observedAt={
              forecast.observed_at
            }
            forecasts={
              forecast.forecasts
            }
          />

          <CurrentConditions
            current={forecast.current}
          />

          <ForecastCards
            forecasts={
              forecast.forecasts
            }
          />

          {performanceLoading &&
          !performance ? (
            <section
              className="
                rounded-[6px]
                border
                border-[var(--color-border)]
                bg-[var(--color-surface)]
                p-6
              "
            >
              <p
                className="
                  text-[12px]
                  text-[var(--color-text-tertiary)]
                "
              >
                Loading forecast reliability…
              </p>
            </section>
          ) : performance ? (
            <ForecastReliability
              performance={performance}
            />
          ) : performanceError ? (
            <section
              className="
                rounded-[6px]
                border
                border-[var(--color-border)]
                bg-[var(--color-surface)]
                p-5
              "
            >
              <p
                className="
                  text-[13px]
                  font-semibold
                "
              >
                Forecast reliability
              </p>

              <p
                className="
                  mt-1
                  text-[12px]
                  leading-5
                  text-[var(--color-text-secondary)]
                "
              >
                Performance information is
                temporarily unavailable.
                Forecasting remains available.
              </p>
            </section>
          ) : null}

          {forecast24 && (
            <ForecastExplanation
              forecast={forecast24}
            />
          )}

          <footer
            className="
              flex
              flex-col
              gap-2
              border-t
              border-[var(--color-border)]
              pt-5
              text-[10px]
              leading-5
              text-[var(--color-text-tertiary)]
              sm:flex-row
              sm:items-center
              sm:justify-between
            "
          >
            <p>
              Source:{' '}
              {forecast.metadata.data_source}
              {' · '}
              {
                forecast.metadata
                  .feature_count
              }{' '}
              features
              {' · '}
              {
                forecast.metadata
                  .history_rows
              }{' '}
              history rows
            </p>

            <p>
              {forecast.metadata.feature_store}
              {' · '}
              {forecast.metadata.model_registry}
            </p>
          </footer>
        </div>
      ) : null}
    </div>
  )
}