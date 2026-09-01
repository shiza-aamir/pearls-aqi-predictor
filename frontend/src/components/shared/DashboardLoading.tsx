export function DashboardLoading() {
  return (
    <div
      className="
        space-y-4
        animate-pulse
      "
      aria-label="Loading air quality data"
    >
      <div
        className="
          h-[230px]
          rounded-[6px]
          border
          border-[var(--color-border)]
          bg-[var(--color-surface)]
        "
      />

      <div
        className="
          h-[250px]
          rounded-[6px]
          border
          border-[var(--color-border)]
          bg-[var(--color-surface)]
        "
      />

      <div
        className="
          grid gap-3
          lg:grid-cols-3
        "
      >
        {[1, 2, 3].map((item) => (
          <div
            key={item}
            className="
              h-[180px]
              rounded-[6px]
              border
              border-[var(--color-border)]
              bg-[var(--color-surface)]
            "
          />
        ))}
      </div>
    </div>
  )
}