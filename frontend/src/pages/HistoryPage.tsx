import {
  RefreshCw,
} from 'lucide-react'

import { CitySelector } from '../components/overview/CitySelector'
import { AQIHistoryChart } from '../components/history/AQIHistoryChart'
import { HistoryAvailability } from '../components/history/HistoryAvailability'
import { HistoryLoading } from '../components/history/HistoryLoading'
import { HistoryRangeSelector } from '../components/history/HistoryRangeSelector'
import { HistoryStatistics } from '../components/history/HistoryStatistics'
import { PollutantHistoryChart } from '../components/history/PollutantHistoryChart'
import { DashboardError } from '../components/shared/DashboardError'
import { useHistoryData } from '../hooks/useHistoryData'
import {
  formatDateTimePKT,
} from '../utils/dates'

export function HistoryPage() {
  const {
    cities,
    selectedCity,
    selectedHours,

    history,

    citiesLoading,
    historyLoading,

    error,

    selectCity,
    selectHours,
    refresh,
  } = useHistoryData()

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
          lg:flex-row
          lg:items-end
          lg:justify-between
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
            Historical observations
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

          {history && (
            <p
              className="
                mt-2
                text-[11px]
                text-[var(--color-text-tertiary)]
              "
            >
              {formatDateTimePKT(
                history.start_time,
              )}
              {' → '}
              {formatDateTimePKT(
                history.end_time,
              )}{' '}
              PKT
            </p>
          )}
        </div>

        <div
          className="
            flex
            flex-col
            gap-2
            sm:flex-row
            sm:items-center
          "
        >
          <HistoryRangeSelector
            value={selectedHours}
            onChange={selectHours}
          />

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
              disabled={historyLoading}
              aria-label="Refresh history"
              title="Refresh history"
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
                  historyLoading
                    ? 'animate-spin'
                    : ''
                }
              />
            </button>

            <CitySelector
              cities={cities}
              selectedCity={
                selectedCity
              }
              loading={
                citiesLoading
              }
              onChange={
                selectCity
              }
            />
          </div>
        </div>
      </div>

      {error && !history ? (
        <DashboardError
          message={error}
          onRetry={refresh}
        />
      ) : historyLoading &&
        !history ? (
        <HistoryLoading />
      ) : history ? (
        <div className="space-y-6">
          {error && (
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
              Refresh failed. Showing
              the most recent available
              history.
            </div>
          )}

          <HistoryAvailability
            requestedHours={
              history.requested_hours
            }
            availableHours={
              history.available_hours
            }
          />

          <AQIHistoryChart
            observations={
              history.observations
            }
          />

          <HistoryStatistics
            statistics={
              history.statistics
            }
          />

          <PollutantHistoryChart
            observations={
              history.observations
            }
          />

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
              {
                history.available_hours
              }{' '}
              AQI-ready hourly
              observations
            </p>

            <p>
              Times displayed in
              Pakistan Standard Time
            </p>
          </footer>
        </div>
      ) : null}
    </div>
  )
}