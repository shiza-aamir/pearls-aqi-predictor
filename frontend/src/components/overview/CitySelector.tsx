import { MapPin } from 'lucide-react'

import type { City } from '../../api/types'

interface CitySelectorProps {
  cities: City[]
  selectedCity: string
  loading: boolean
  onChange: (city: string) => void
}

export function CitySelector({
  cities,
  selectedCity,
  loading,
  onChange,
}: CitySelectorProps) {
  return (
    <label className="relative block">
      <span className="sr-only">
        Select city
      </span>

      <MapPin
        aria-hidden="true"
        size={16}
        strokeWidth={1.8}
        className="
          pointer-events-none
          absolute left-3 top-1/2
          -translate-y-1/2
          text-[var(--color-text-tertiary)]
        "
      />

      <select
        value={selectedCity}
        disabled={loading}
        onChange={(event) =>
          onChange(event.target.value)
        }
        className="
          min-w-[190px]
          appearance-none
          rounded-[5px]
          border
          border-[var(--color-border-strong)]
          bg-[var(--color-surface)]
          py-2.5 pl-9 pr-9
          text-[13px]
          font-medium
          text-[var(--color-text-primary)]
          shadow-[0_1px_2px_rgba(28,36,48,0.03)]
          outline-none
          transition-colors
          hover:border-[var(--color-text-tertiary)]
          disabled:cursor-not-allowed
          disabled:opacity-60
        "
      >
        {cities.length === 0 ? (
          <option value={selectedCity}>
            {selectedCity}
          </option>
        ) : (
          cities.map((city) => (
            <option
              key={city.name}
              value={city.name}
            >
              {city.name}
            </option>
          ))
        )}
      </select>

      <span
        aria-hidden="true"
        className="
          pointer-events-none
          absolute right-3 top-1/2
          -translate-y-[55%]
          text-[10px]
          text-[var(--color-text-tertiary)]
        "
      >
        ▼
      </span>
    </label>
  )
}