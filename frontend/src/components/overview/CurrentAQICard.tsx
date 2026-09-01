import { ShieldAlert } from 'lucide-react'

import type {
  CurrentAirQuality,
} from '../../api/types'
import {
  getAQICategoryPresentation,
} from '../../utils/aqi'
import { formatAQI } from '../../utils/formatters'
import { AQIStatus } from '../shared/AQIStatus'

interface CurrentAQICardProps {
  city: string
  current: CurrentAirQuality
}

interface AQIBand {
  minimum: number
  maximum: number
  range: string
  category: string
  shortLabel: string
  explanation: string
}

const AQI_BANDS: AQIBand[] = [
  {
    minimum: 0,
    maximum: 50,
    range: '0–50',
    category: 'Good',
    shortLabel: 'Good',
    explanation:
      'Air quality is satisfactory. Normal outdoor activity is appropriate.',
  },
  {
    minimum: 51,
    maximum: 100,
    range: '51–100',
    category: 'Moderate',
    shortLabel: 'Moderate',
    explanation:
      'Air quality is generally acceptable for most people.',
  },
  {
    minimum: 101,
    maximum: 150,
    range: '101–150',
    category: 'Unhealthy for Sensitive Groups',
    shortLabel: 'Sensitive Groups',
    explanation:
      'Most people can continue normal activity; sensitive people should reduce prolonged or strenuous outdoor activity.',
  },
  {
    minimum: 151,
    maximum: 200,
    range: '151–200',
    category: 'Unhealthy',
    shortLabel: 'Unhealthy',
    explanation:
      'Health effects may affect everyone, with greater risk for sensitive groups.',
  },
  {
    minimum: 201,
    maximum: 300,
    range: '201–300',
    category: 'Very Unhealthy',
    shortLabel: 'Very Unhealthy',
    explanation:
      'Health risk is increased for everyone. Limit prolonged outdoor activity.',
  },
  {
    minimum: 301,
    maximum: 500,
    range: '301–500',
    category: 'Hazardous',
    shortLabel: 'Hazardous',
    explanation:
      'Serious health risk is possible. Avoid strenuous outdoor activity.',
  },
]

export function CurrentAQICard({
  city,
  current,
}: CurrentAQICardProps) {
  const presentation =
    getAQICategoryPresentation(
      current.category,
    )

  const currentAQI = Math.round(
    current.aqi,
  )

  const activeBand =
    AQI_BANDS.find(
      (band) =>
        currentAQI >= band.minimum &&
        currentAQI <= band.maximum,
    ) ??
    AQI_BANDS[AQI_BANDS.length - 1]

  return (
    <section
      className="
        overflow-hidden
        rounded-[6px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
      "
      aria-labelledby="current-aqi-heading"
    >
      <div
        className="h-[4px]"
        style={{
          backgroundColor: presentation.color,
        }}
      />

      <div className="p-6 md:p-8">
        <div
          className="
            flex
            flex-col
            gap-8
            lg:flex-row
            lg:items-end
            lg:justify-between
          "
        >
          <div>
            <p
              id="current-aqi-heading"
              className="
                text-[12px]
                font-semibold
                uppercase
                tracking-[0.12em]
                text-[var(--color-text-tertiary)]
              "
            >
              Current AQI · {city}
            </p>

            <div
              className="
                mt-4
                flex
                items-end
                gap-5
              "
            >
              <span
                className="
                  font-display
                  text-[82px]
                  font-medium
                  leading-[0.82]
                  tracking-[-0.055em]
                  md:text-[104px]
                "
              >
                {formatAQI(current.aqi)}
              </span>

              <div className="pb-1 md:pb-2">
                <AQIStatus
                  category={current.category}
                />

                <p
                  className="
                    mt-1
                    text-[11px]
                    uppercase
                    tracking-[0.08em]
                    text-[var(--color-text-tertiary)]
                  "
                >
                  US AQI
                </p>
              </div>
            </div>
          </div>

          <div
            className="
              max-w-xl
              border-t
              border-[var(--color-border)]
              pt-5
              lg:border-l
              lg:border-t-0
              lg:pl-7
              lg:pt-0
            "
          >
            <div
              className="
                flex
                items-start
                gap-3
              "
            >
              <ShieldAlert
                aria-hidden="true"
                size={18}
                strokeWidth={1.7}
                className="
                  mt-[2px]
                  shrink-0
                  text-[var(--color-text-secondary)]
                "
              />

              <div>
                <p
                  className="
                    text-[13px]
                    font-semibold
                    text-[var(--color-text-primary)]
                  "
                >
                  {current.alert.level}
                </p>

                <p
                  className="
                    mt-1.5
                    text-[13px]
                    leading-6
                    text-[var(--color-text-secondary)]
                  "
                >
                  {current.alert.message}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div
          className="
            mt-7
            border-t
            border-[var(--color-border)]
            pt-5
            md:mt-8
          "
        >
          <div
            className="
              mb-3
              flex
              items-center
              justify-between
              gap-4
            "
          >
            <p
              className="
                text-[10px]
                font-semibold
                uppercase
                tracking-[0.1em]
                text-[var(--color-text-tertiary)]
              "
            >
              AQI health scale
            </p>

            <p
              className="
                hidden
                text-[10px]
                text-[var(--color-text-tertiary)]
                sm:block
              "
            >
              Lower AQI indicates cleaner air
            </p>
          </div>

          <div
            className="
              grid
              grid-cols-3
              gap-1.5
              sm:grid-cols-6
            "
          >
            {AQI_BANDS.map((band) => {
              const bandPresentation =
                getAQICategoryPresentation(
                  band.category,
                )

              const isCurrent =
                band.category ===
                activeBand.category

              return (
                <div
                  key={band.category}
                  className={`
                    relative
                    rounded-[4px]
                    border
                    px-2
                    pb-2.5
                    pt-3
                    ${
                      isCurrent
                        ? 'border-[var(--color-border-strong)] bg-[var(--color-surface-sunken)]'
                        : 'border-transparent'
                    }
                  `}
                >
                  <div
                    className="
                      absolute
                      inset-x-0
                      top-0
                      h-[3px]
                      rounded-t-[3px]
                    "
                    style={{
                      backgroundColor:
                        bandPresentation.color,
                    }}
                  />

                  <p
                    className="
                      font-mono
                      text-[9px]
                      text-[var(--color-text-tertiary)]
                    "
                  >
                    {band.range}
                  </p>

                  <p
                    className="
                      mt-1
                      text-[10px]
                      font-semibold
                      leading-[1.35]
                    "
                    style={{
                      color:
                        bandPresentation.color,
                    }}
                  >
                    {band.shortLabel}
                  </p>

                  {isCurrent && (
                    <p
                      className="
                        mt-1.5
                        text-[8px]
                        font-semibold
                        uppercase
                        tracking-[0.08em]
                        text-[var(--color-accent-strong)]
                      "
                    >
                      Current · {currentAQI}
                    </p>
                  )}
                </div>
              )
            })}
          </div>

          <div
            className="
              mt-4
              flex
              flex-col
              gap-1
              sm:flex-row
              sm:items-baseline
              sm:gap-2
            "
          >
            <p
              className="
                text-[11px]
                font-semibold
                text-[var(--color-text-primary)]
              "
            >
              {currentAQI} falls in{' '}
              {activeBand.category}.
            </p>

            <p
              className="
                text-[11px]
                leading-5
                text-[var(--color-text-secondary)]
              "
            >
              {activeBand.explanation}
            </p>
          </div>

          <p
            className="
              mt-2
              text-[9px]
              leading-4
              text-[var(--color-text-tertiary)]
            "
          >
            AQI represents population-level
            air quality risk; individual
            sensitivity can vary.
          </p>
        </div>
      </div>
    </section>
  )
}