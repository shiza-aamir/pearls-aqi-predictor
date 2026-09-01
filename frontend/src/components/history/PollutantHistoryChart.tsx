import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type {
  HistoryObservation,
} from '../../api/types'
import {
  formatDateTimePKT,
  formatTimePKT,
} from '../../utils/dates'
import {
  formatDecimal,
} from '../../utils/formatters'

interface PollutantHistoryChartProps {
  observations: HistoryObservation[]
}

interface ChartPoint {
  timestamp: string
  pm2_5: number
  pm10: number
}

interface TooltipPayloadItem {
  dataKey: string
  value: number
  payload: ChartPoint
}

interface PollutantTooltipProps {
  active?: boolean
  payload?: TooltipPayloadItem[]
}

function PollutantTooltip({
  active,
  payload,
}: PollutantTooltipProps) {
  if (
    !active ||
    !payload ||
    payload.length === 0
  ) {
    return null
  }

  const point = payload[0].payload

  const pm25 =
    payload.find(
      (item) => item.dataKey === 'pm2_5',
    )?.value ?? point.pm2_5

  const pm10 =
    payload.find(
      (item) => item.dataKey === 'pm10',
    )?.value ?? point.pm10

  return (
    <div
      className="
        w-[190px]
        rounded-[5px]
        border
        border-[var(--color-border)]
        bg-[var(--color-surface)]
        px-3
        py-2.5
        shadow-[0_4px_16px_rgba(28,36,48,0.08)]
        sm:w-[210px]
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
          mt-2.5
          space-y-1.5
          sm:mt-3
          sm:space-y-2
        "
      >
        <div
          className="
            flex
            items-center
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
            PM2.5
          </span>

          <span
            className="
              font-mono
              text-[11px]
              sm:text-[12px]
            "
          >
            {formatDecimal(
              pm25,
              1,
            )}{' '}
            µg/m³
          </span>
        </div>

        <div
          className="
            flex
            items-center
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
            PM10
          </span>

          <span
            className="
              font-mono
              text-[11px]
              sm:text-[12px]
            "
          >
            {formatDecimal(
              pm10,
              1,
            )}{' '}
            µg/m³
          </span>
        </div>
      </div>
    </div>
  )
}

export function PollutantHistoryChart({
  observations,
}: PollutantHistoryChartProps) {
  const data: ChartPoint[] =
    observations.map((observation) => ({
      timestamp: observation.timestamp,
      pm2_5: observation.pm2_5,
      pm10: observation.pm10,
    }))

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
      aria-labelledby="pollutant-history-heading"
    >
      <div>
        <h2
          id="pollutant-history-heading"
          className="
            text-[14px]
            font-semibold
          "
        >
          Pollutant trends
        </h2>

        <p
          className="
            mt-1
            text-[11px]
            text-[var(--color-text-tertiary)]
          "
        >
          Hourly particulate matter
          concentrations
        </p>
      </div>

      <div
        className="
          mt-5
          flex
          items-center
          gap-5
        "
      >
        <div
          className="
            flex
            items-center
            gap-2
            text-[10px]
            text-[var(--color-text-secondary)]
          "
        >
          <span
            className="
              h-[2px]
              w-5
              bg-[var(--color-accent-strong)]
            "
          />
          PM2.5
        </div>

        <div
          className="
            flex
            items-center
            gap-2
            text-[10px]
            text-[var(--color-text-secondary)]
          "
        >
          <span
            className="
              h-[2px]
              w-5
              bg-[var(--color-text-tertiary)]
            "
          />
          PM10
        </div>
      </div>

      <div
        className="
          mt-4
          h-[310px]
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
              left: -10,
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
              tick={{
                fontSize: 10,
                fill: 'var(--color-text-tertiary)',
              }}
              tickLine={false}
              axisLine={false}
              width={44}
              unit=""
            />

            <Tooltip
              content={
                <PollutantTooltip />
              }
              wrapperStyle={{
                outline: 'none',
              }}
            />

            <Line
              type="monotone"
              dataKey="pm2_5"
              name="PM2.5"
              stroke="var(--color-accent-strong)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />

            <Line
              type="monotone"
              dataKey="pm10"
              name="PM10"
              stroke="var(--color-text-tertiary)"
              strokeWidth={1.7}
              strokeDasharray="6 4"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p
        className="
          mt-3
          text-[9px]
          text-[var(--color-text-tertiary)]
        "
      >
        Concentrations shown in µg/m³.
      </p>
    </section>
  )
}