export function HistoryLoading() {
  return (
    <div
      className="
        space-y-5
        animate-pulse
      "
      aria-label="Loading historical air quality data"
    >
      <div
        className="
          h-[410px]
          rounded-[6px]
          border
          border-[var(--color-border)]
          bg-[var(--color-surface)]
        "
      />

      <div
        className="
          h-[160px]
          rounded-[6px]
          border
          border-[var(--color-border)]
          bg-[var(--color-surface)]
        "
      />

      <div
        className="
          h-[380px]
          rounded-[6px]
          border
          border-[var(--color-border)]
          bg-[var(--color-surface)]
        "
      />
    </div>
  )
}