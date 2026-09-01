import {
  getAQICategoryPresentation,
} from '../../utils/aqi'

interface AQIStatusProps {
  category: string
  compact?: boolean
}

export function AQIStatus({
  category,
  compact = false,
}: AQIStatusProps) {
  const presentation =
    getAQICategoryPresentation(category)

  return (
    <span
      className="
        inline-flex
        items-center gap-2
        font-medium
      "
      style={{
        color: presentation.color,
      }}
    >
      <span
        aria-hidden="true"
        className="
          inline-block
          h-2 w-2
          shrink-0 rounded-full
        "
        style={{
          backgroundColor: presentation.color,
        }}
      />

      <span
        className={
          compact
            ? 'text-[12px]'
            : 'text-[13px]'
        }
      >
        {compact
          ? presentation.shortLabel
          : presentation.label}
      </span>
    </span>
  )
}