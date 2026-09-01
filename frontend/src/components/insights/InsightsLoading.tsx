export function InsightsLoading() {
  return (
    <div
      className="
        space-y-5
        animate-pulse
      "
      aria-label="Loading model insights"
    >
      <div
        className="
          grid
          gap-3
          lg:grid-cols-3
        "
      >
        {[1, 2, 3].map((item) => (
          <div
            key={item}
            className="
              h-[240px]
              rounded-[6px]
              border
              border-[var(--color-border)]
              bg-[var(--color-surface)]
            "
          />
        ))}
      </div>

      <div
        className="
          h-[290px]
          rounded-[6px]
          border
          border-[var(--color-border)]
          bg-[var(--color-surface)]
        "
      />

      <div
        className="
          h-[300px]
          rounded-[6px]
          border
          border-[var(--color-border)]
          bg-[var(--color-surface)]
        "
      />
    </div>
  )
}