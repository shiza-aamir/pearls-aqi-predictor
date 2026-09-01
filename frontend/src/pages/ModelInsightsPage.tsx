import {
  RefreshCw,
} from 'lucide-react'

import { CitySelector } from '../components/overview/CitySelector'
import { DashboardError } from '../components/shared/DashboardError'
import { EvaluationProtocol } from '../components/insights/EvaluationProtocol'
import { ExplanationExplorer } from '../components/insights/ExplanationExplorer'
import { HoldoutEvaluationTable } from '../components/insights/HoldoutEvaluationTable'
import { InsightsLoading } from '../components/insights/InsightsLoading'
import { LiveModelMonitoring } from '../components/insights/LiveModelMonitoring'
import { ModelCandidates } from '../components/insights/ModelCandidates'
import { ProductionModels } from '../components/insights/ProductionModels'
import { useModelInsightsData } from '../hooks/useModelInsightsData'

export function ModelInsightsPage() {
  const {
    cities,
    selectedCity,

    models,
    performance,
    forecast,

    citiesLoading,
    modelsLoading,
    performanceLoading,
    forecastLoading,

    modelsError,
    performanceError,
    forecastError,

    selectCity,
    refresh,
  } = useModelInsightsData()

  const initialLoading =
    modelsLoading &&
    performanceLoading &&
    forecastLoading &&
    !models &&
    !performance &&
    !forecast

  const refreshing =
    modelsLoading ||
    performanceLoading ||
    forecastLoading

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
            Model transparency
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
            Model Insights
          </h1>

          <p
            className="
              mt-2
              max-w-2xl
              text-[11px]
              leading-5
              text-[var(--color-text-tertiary)]
            "
          >
            Evaluation, explainability and
            production monitoring for the
            forecasting system.
          </p>
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
            disabled={refreshing}
            aria-label="Refresh model insights"
            title="Refresh model insights"
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
                refreshing
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

      {initialLoading ? (
        <InsightsLoading />
      ) : modelsError &&
        !models &&
        performanceError &&
        !performance &&
        forecastError &&
        !forecast ? (
        <DashboardError
          message="Model insight information is currently unavailable."
          onRetry={refresh}
        />
      ) : (
        <div className="space-y-7">
          {models ? (
            <>
              <ProductionModels
                models={
                  models.production_models
                }
              />

              <div
                className="
                  grid
                  gap-6
                  xl:grid-cols-[1fr_2fr]
                "
              >
                <ModelCandidates
                  candidates={
                    models.evaluated_candidates
                  }
                />

                <EvaluationProtocol
                  evaluation={
                    models.evaluation
                  }
                />
              </div>
            </>
          ) : modelsError ? (
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
                Model registry unavailable
              </p>

              <p
                className="
                  mt-1
                  text-[11px]
                  text-[var(--color-text-secondary)]
                "
              >
                Production registry
                information could not be
                loaded.
              </p>
            </section>
          ) : null}

          {forecastLoading &&
          !forecast ? (
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
                  text-[11px]
                  text-[var(--color-text-tertiary)]
                "
              >
                Loading forecast
                explanations…
              </p>
            </section>
          ) : forecast ? (
            <ExplanationExplorer
              forecast={forecast}
            />
          ) : forecastError ? (
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
                Forecast explanations
                unavailable
              </p>

              <p
                className="
                  mt-1
                  text-[11px]
                  leading-5
                  text-[var(--color-text-secondary)]
                "
              >
                Model evaluation remains
                available, but current SHAP
                explanations could not be
                loaded for {selectedCity}.
              </p>
            </section>
          ) : null}

          {performance ? (
            <>
              <HoldoutEvaluationTable
                rows={
                  performance.holdout
                }
              />

              <LiveModelMonitoring
                performance={
                  performance
                }
              />
            </>
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
                Performance data
                unavailable
              </p>

              <p
                className="
                  mt-1
                  text-[11px]
                  leading-5
                  text-[var(--color-text-secondary)]
                "
              >
                Technical model information
                remains available, but
                performance metrics could not
                be loaded for {selectedCity}.
              </p>
            </section>
          ) : null}

          <footer
            className="
              border-t
              border-[var(--color-border)]
              pt-5
              text-[9px]
              leading-5
              text-[var(--color-text-tertiary)]
            "
          >
            Final holdout results are
            preserved for model assessment
            and are not used for model
            reselection. SHAP contributions
            explain model behaviour rather
            than causal effects. Live
            performance is evaluated
            separately from matured
            production forecasts.
          </footer>
        </div>
      )}
    </div>
  )
}