interface ModelCandidatesProps {
  candidates: string[]
}

export function ModelCandidates({
  candidates,
}: ModelCandidatesProps) {
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
      aria-labelledby="model-candidates-heading"
    >
      <div>
        <h2
          id="model-candidates-heading"
          className="
            text-[14px]
            font-semibold
          "
        >
          Evaluated candidates
        </h2>

        <p
          className="
            mt-1
            text-[11px]
            leading-5
            text-[var(--color-text-tertiary)]
          "
        >
          Candidate models compared through
          time-aware validation before
          production selection.
        </p>
      </div>

      <div
        className="
          mt-6
          flex
          flex-wrap
          gap-2
        "
      >
        {candidates.map((candidate) => (
          <span
            key={candidate}
            className="
              rounded-[5px]
              border
              border-[var(--color-border)]
              bg-[var(--color-surface-sunken)]
              px-3
              py-2
              text-[11px]
              font-medium
              text-[var(--color-text-secondary)]
            "
          >
            {candidate}
          </span>
        ))}
      </div>

      <p
        className="
          mt-5
          border-t
          border-[var(--color-border)]
          pt-4
          text-[9px]
          leading-5
          text-[var(--color-text-tertiary)]
        "
      >
        These models represent the evaluation
        set used during experimentation. They
        are not all serving forecasts in
        production.
      </p>
    </section>
  )
}