import type {
  ForecastItem,
} from '../../api/types'
import {
  formatDecimal,
  formatSignedNumber,
} from '../../utils/formatters'

interface ForecastExplanationProps {
  forecast: ForecastItem
}

export function ForecastExplanation({
  forecast,
}: ForecastExplanationProps) {
  const contributions =
    forecast.explanation.top_features

  const maxMagnitude = Math.max(
    ...contributions.map((item) =>
      Math.abs(item.contribution),
    ),
    1,
  )

  return (
    <section
      className="
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
        p-5 md:p-6
      "
      aria-labelledby="explanation-heading"
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
            id="explanation-heading"
            className="
              text-[14px]
              font-semibold
            "
          >
            What is influencing the next
            24 hours?
          </h2>

          <p
            className="
              mt-1 text-[12px]
              text-[var(--color-text-tertiary)]
            "
          >
            Largest SHAP contributions to
            the 24-hour forecast
          </p>
        </div>

        <span
          className="
            font-mono text-[10px]
            text-[var(--color-text-tertiary)]
          "
        >
          base{' '}
          {formatDecimal(
            forecast.explanation.base_value,
            1,
          )}
        </span>
      </div>

      <div className="mt-7 space-y-5">
        {contributions.map((item) => {
          const magnitude =
            (Math.abs(item.contribution) /
              maxMagnitude) *
            100

          const increasing =
            item.direction === 'increase'

          return (
            <div key={item.feature}>
              <div
                className="
                  flex items-start
                  justify-between gap-4
                "
              >
                <div className="min-w-0">
                  <p
                    className="
                      truncate
                      text-[12px]
                      font-medium
                    "
                    title={item.display_name}
                  >
                    {item.display_name}
                  </p>

                  <p
                    className="
                      font-mono mt-1
                      text-[9px]
                      text-[var(--color-text-tertiary)]
                    "
                  >
                    value{' '}
                    {formatDecimal(
                      item.feature_value,
                      2,
                    )}
                  </p>
                </div>

                <span
                  className="
                    font-mono shrink-0
                    text-[11px]
                    font-medium
                  "
                  style={{
                    color: increasing
                      ? '#B54B3F'
                      : '#3F6D5C',
                  }}
                >
                  {formatSignedNumber(
                    item.contribution,
                    2,
                  )}
                </span>
              </div>

              <div
                className="
                  mt-2 h-[4px]
                  overflow-hidden
                  rounded-full
                  bg-[var(--color-surface-sunken)]
                "
              >
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${magnitude}%`,
                    backgroundColor: increasing
                      ? '#B54B3F'
                      : '#3F6D5C',
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <p
        className="
          mt-6 border-t
          border-[var(--color-border)]
          pt-4
          text-[10px]
          leading-5
          text-[var(--color-text-tertiary)]
        "
      >
        Positive contributions push the
        forecast higher; negative
        contributions push it lower. These
        values explain model behavior and
        should not be interpreted as causal
        effects.
      </p>
    </section>
  )
}