import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type {
  HistoryObservation,
} from '../../api/types'
import {
  getAQICategoryPresentation,
} from '../../utils/aqi'
import {
  formatDateTimePKT,
  formatTimePKT,
} from '../../utils/dates'

interface AQIHistoryChartProps {
  observations: HistoryObservation[]
}

interface ChartPoint {
  timestamp: string
  aqi: number
  category: string
}

interface TooltipPayloadItem {
  payload: ChartPoint
}

interface AQITooltipProps {
  active?: boolean
  payload?: TooltipPayloadItem[]
}

const AQI_THRESHOLDS = [
  {
    value: 50,
    label: 'Good',
    color: 'var(--aqi-good)',
  },
  {
    value: 100,
    label: 'Moderate',
    color: 'var(--aqi-moderate)',
  },
  {
    value: 150,
    label: 'Sensitive Groups',
    color: 'var(--aqi-usg)',
  },
  {
    value: 200,
    label: 'Unhealthy',
    color: 'var(--aqi-unhealthy)',
  },
  {
    value: 300,
    label: 'Very Unhealthy',
    color: 'var(--aqi-very-unhealthy)',
  },
]

function AQITooltip({
  active,
  payload,
}: AQITooltipProps) {
  if (
    !active ||
    !payload ||
    payload.length === 0
  ) {
    return null
  }

  const point = payload[0].payload

  const presentation =
    getAQICategoryPresentation(
      point.category,
    )

  return (
    <div
      className="
        w-[176px]
        rounded-[5px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
        px-3
        py-2.5
        shadow-[0_4px_16px_rgba(28,36,48,0.08)]
        sm:w-[190px]
        sm:px-4
        sm:py-3
      "
    >
      <p
        className="
          text-[9px]
          text-[var(--color-text-tertiary)]
          sm:text-[10px]
        "
      >
        {formatDateTimePKT(
          point.timestamp,
        )}
      </p>

      <div
        className="
          mt-2
          flex
          items-baseline
          justify-between
          gap-4
        "
      >
        <span
          className="
            text-[10px]
            font-medium
            sm:text-[11px]
          "
        >
          AQI
        </span>

        <span
          className="
            font-mono
            text-[13px]
            font-medium
            sm:text-[14px]
          "
        >
          {Math.round(point.aqi)}
        </span>
      </div>

      <p
        className="
          mt-1
          text-[9px]
          font-medium
          leading-4
          sm:text-[10px]
        "
        style={{
          color: presentation.color,
        }}
      >
        {presentation.label}
      </p>
    </div>
  )
}

function getAdaptiveDomain(
  observations: HistoryObservation[],
): [number, number] {
  if (observations.length === 0) {
    return [0, 200]
  }

  const values = observations.map(
    (observation) => observation.aqi,
  )

  const minimum = Math.min(...values)
  const maximum = Math.max(...values)

  const observedRange = Math.max(
    maximum - minimum,
    1,
  )

  const padding = Math.max(
    observedRange * 0.35,
    10,
  )

  let lower = Math.max(
    0,
    Math.floor(
      (minimum - padding) / 10,
    ) * 10,
  )

  let upper = Math.min(
    500,
    Math.ceil(
      (maximum + padding) / 10,
    ) * 10,
  )

  const relevantThresholds =
    AQI_THRESHOLDS.filter(
      (threshold) =>
        threshold.value >= lower - 10 &&
        threshold.value <= upper + 10,
    )

  for (
    const threshold of relevantThresholds
  ) {
    lower = Math.min(
      lower,
      Math.max(
        0,
        threshold.value - 10,
      ),
    )

    upper = Math.max(
      upper,
      Math.min(
        500,
        threshold.value + 10,
      ),
    )
  }

  if (upper - lower < 40) {
    const center =
      (upper + lower) / 2

    lower = Math.max(
      0,
      Math.floor(
        (center - 20) / 10,
      ) * 10,
    )

    upper = Math.min(
      500,
      Math.ceil(
        (center + 20) / 10,
      ) * 10,
    )
  }

  if (upper <= lower) {
    upper = Math.min(
      500,
      lower + 40,
    )
  }

  return [lower, upper]
}

function getYAxisTicks(
  lower: number,
  upper: number,
): number[] {
  const span = upper - lower

  let step = 10

  if (span > 250) {
    step = 100
  } else if (span > 120) {
    step = 50
  } else if (span > 70) {
    step = 20
  }

  const ticks: number[] = []

  const start =
    Math.ceil(lower / step) * step

  for (
    let value = start;
    value <= upper;
    value += step
  ) {
    ticks.push(value)
  }

  if (ticks.length < 3) {
    return [
      lower,
      Math.round(
        (lower + upper) / 2,
      ),
      upper,
    ]
  }

  return ticks
}

export function AQIHistoryChart({
  observations,
}: AQIHistoryChartProps) {
  const data: ChartPoint[] =
    observations.map((observation) => ({
      timestamp: observation.timestamp,
      aqi: observation.aqi,
      category: observation.category,
    }))

  const [domainMinimum, domainMaximum] =
    getAdaptiveDomain(observations)

  const ticks = getYAxisTicks(
    domainMinimum,
    domainMaximum,
  )

  const visibleThresholds =
    AQI_THRESHOLDS.filter(
      (threshold) =>
        threshold.value >= domainMinimum &&
        threshold.value <= domainMaximum,
    )

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
      aria-labelledby="aqi-history-chart-heading"
    >
      <div
        className="
          flex
          flex-col
          gap-2
          sm:flex-row
          sm:items-end
          sm:justify-between
        "
      >
        <div>
          <h2
            id="aqi-history-chart-heading"
            className="
              text-[14px]
              font-semibold
            "
          >
            Air quality over time
          </h2>

          <p
            className="
              mt-1
              text-[11px]
              text-[var(--color-text-tertiary)]
            "
          >
            Hourly US AQI observations
          </p>
        </div>

        <div
          className="
            flex
            items-center
            gap-4
          "
        >
          <span
            className="
              text-[9px]
              text-[var(--color-text-tertiary)]
            "
          >
            Adaptive scale
          </span>

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
      </div>

      <div
        className="
          mt-6
          h-[340px]
          w-full
        "
      >
        <ResponsiveContainer
          width="100%"
          height="100%"
        >
          <LineChart
            data={data}
            margin={{
              top: 10,
              right: 14,
              left: -8,
              bottom: 0,
            }}
          >
            <CartesianGrid
              stroke="var(--color-border)"
              strokeDasharray="3 5"
              vertical={false}
            />

            <XAxis
              dataKey="timestamp"
              tickFormatter={
                formatTimePKT
              }
              tick={{
                fontSize: 10,
                fill: 'var(--color-text-tertiary)',
              }}
              tickLine={false}
              axisLine={false}
              minTickGap={38}
            />

            <YAxis
              domain={[
                domainMinimum,
                domainMaximum,
              ]}
              ticks={ticks}
              allowDataOverflow
              tick={{
                fontSize: 10,
                fill: 'var(--color-text-tertiary)',
              }}
              tickLine={false}
              axisLine={false}
              width={45}
            />

            {visibleThresholds.map(
              (threshold) => (
                <ReferenceLine
                  key={threshold.value}
                  y={threshold.value}
                  stroke={threshold.color}
                  strokeOpacity={0.32}
                  strokeDasharray="4 4"
                />
              ),
            )}

            <Tooltip
              content={<AQITooltip />}
              wrapperStyle={{
                outline: 'none',
              }}
            />

            <Line
              type="monotone"
              dataKey="aqi"
              stroke="var(--color-accent-strong)"
              strokeWidth={2.2}
              dot={false}
              activeDot={{
                r: 4,
                strokeWidth: 2,
                stroke:
                  'var(--color-surface)',
                fill:
                  'var(--color-accent-strong)',
              }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div
        className="
          mt-4
          flex
          flex-wrap
          gap-x-5
          gap-y-2
          border-t
          border-[var(--color-border)]
          pt-4
        "
      >
        {AQI_THRESHOLDS.map(
          (item) => (
            <div
              key={item.label}
              className="
                flex
                items-center
                gap-2
                text-[9px]
                text-[var(--color-text-tertiary)]
              "
            >
              <span
                className="
                  h-2
                  w-2
                  rounded-full
                "
                style={{
                  backgroundColor:
                    item.color,
                }}
              />

              {item.label}
            </div>
          ),
        )}

        <div
          className="
            ml-auto
            text-[9px]
            text-[var(--color-text-tertiary)]
          "
        >
          Visible range:{' '}
          {domainMinimum}–{domainMaximum}
        </div>
      </div>

      <p
        className="
          mt-3
          text-[9px]
          leading-5
          text-[var(--color-text-tertiary)]
        "
      >
        The vertical scale adapts to the
        selected observations with additional
        padding. AQI category thresholds remain
        visible when they fall inside the
        displayed range.
      </p>
    </section>
  )
}