import {
  Droplets,
  Gauge,
  Thermometer,
  Wind,
} from 'lucide-react'

import type {
  CurrentAirQuality,
} from '../../api/types'
import {
  formatDecimal,
} from '../../utils/formatters'

interface CurrentConditionsProps {
  current: CurrentAirQuality
}

interface ConditionItemProps {
  label: string
  value: string
  icon: React.ReactNode
}

function ConditionItem({
  label,
  value,
  icon,
}: ConditionItemProps) {
  return (
    <div
      className="
        min-w-0
        border-[var(--color-border)]
        lg:border-r
        lg:pr-5
        lg:last:border-r-0
        lg:last:pr-0
      "
    >
      <div
        className="
          flex items-center gap-2
          text-[var(--color-text-tertiary)]
        "
      >
        {icon}

        <span
          className="
            text-[11px]
            font-semibold
            uppercase
            tracking-[0.08em]
          "
        >
          {label}
        </span>
      </div>

      <p
        className="
          font-mono
          mt-3
          whitespace-nowrap
          text-[18px]
          font-medium
          text-[var(--color-text-primary)]
        "
      >
        {value}
      </p>
    </div>
  )
}

export function CurrentConditions({
  current,
}: CurrentConditionsProps) {
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
      aria-labelledby="conditions-heading"
    >
      <div
        className="
          mb-6
          flex
          items-baseline
          justify-between
          gap-4
        "
      >
        <h2
          id="conditions-heading"
          className="
            text-[14px]
            font-semibold
          "
        >
          Current conditions
        </h2>

        <span
          className="
            text-[11px]
            text-[var(--color-text-tertiary)]
          "
        >
          Live observation
        </span>
      </div>

      <div
        className="
          grid
          grid-cols-2
          gap-x-6
          gap-y-7
          sm:grid-cols-3
          lg:grid-cols-5
          lg:gap-y-0
        "
      >
        <ConditionItem
          label="PM2.5"
          value={`${formatDecimal(
            current.pollutants.pm2_5,
            1,
          )} µg/m³`}
          icon={
            <Gauge
              size={15}
              strokeWidth={1.7}
            />
          }
        />

        <ConditionItem
          label="PM10"
          value={`${formatDecimal(
            current.pollutants.pm10,
            1,
          )} µg/m³`}
          icon={
            <Gauge
              size={15}
              strokeWidth={1.7}
            />
          }
        />

        <ConditionItem
          label="Temperature"
          value={`${formatDecimal(
            current.weather.temperature_c,
            1,
          )} °C`}
          icon={
            <Thermometer
              size={15}
              strokeWidth={1.7}
            />
          }
        />

        <ConditionItem
          label="Humidity"
          value={`${formatDecimal(
            current.weather.humidity_percent,
            0,
          )}%`}
          icon={
            <Droplets
              size={15}
              strokeWidth={1.7}
            />
          }
        />

        <ConditionItem
          label="Wind"
          value={`${formatDecimal(
            current.weather.wind_speed_ms,
            1,
          )} m/s`}
          icon={
            <Wind
              size={15}
              strokeWidth={1.7}
            />
          }
        />
      </div>
    </section>
  )
}