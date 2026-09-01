import {
  CalendarRange,
  CheckCircle2,
  Database,
} from 'lucide-react'

import type {
  ModelEvaluation,
} from '../../api/types'

interface EvaluationProtocolProps {
  evaluation: ModelEvaluation
}

function formatDatasetTimestamp(
  value: string,
): string {
  const parsed = new Date(value)

  if (Number.isNaN(parsed.getTime())) {
    return value
  }

  return new Intl.DateTimeFormat(
    'en-GB',
    {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
      timeZone: 'UTC',
    },
  ).format(parsed)
}

interface MetricProps {
  label: string
  value: string
}

function Metric({
  label,
  value,
}: MetricProps) {
  return (
    <div>
      <p
        className="
          text-[9px]
          font-semibold
          uppercase
          tracking-[0.08em]
          text-[var(--color-text-tertiary)]
        "
      >
        {label}
      </p>

      <p
        className="
          mt-1
          text-[12px]
          font-semibold
          text-[var(--color-text-primary)]
        "
      >
        {value}
      </p>
    </div>
  )
}

export function EvaluationProtocol({
  evaluation,
}: EvaluationProtocolProps) {
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
      aria-labelledby="evaluation-protocol-heading"
    >
      <div
        className="
          flex
          flex-col
          gap-4
          sm:flex-row
          sm:items-start
          sm:justify-between
        "
      >
        <div>
          <h2
            id="evaluation-protocol-heading"
            className="
              text-[14px]
              font-semibold
            "
          >
            Evaluation protocol
          </h2>

          <p
            className="
              mt-1
              text-[11px]
              leading-5
              text-[var(--color-text-tertiary)]
            "
          >
            Final model assessment after
            production selection was frozen.
          </p>
        </div>

        {evaluation.selection_frozen_before_test && (
          <div
            className="
              inline-flex
              items-center
              gap-2
              self-start
              rounded-[5px]
              bg-[var(--color-accent-soft)]
              px-3
              py-2
              text-[9px]
              font-semibold
              uppercase
              tracking-[0.07em]
              text-[var(--color-accent-strong)]
            "
          >
            <CheckCircle2
              size={13}
              strokeWidth={1.8}
            />

            Selection frozen before test
          </div>
        )}
      </div>

      <div
        className="
          mt-6
          grid
          gap-5
          border-t
          border-[var(--color-border)]
          pt-5
          sm:grid-cols-2
          xl:grid-cols-4
        "
      >
        <Metric
          label="Selection metric"
          value={evaluation.selection_metric}
        />

        <Metric
          label="Training rows"
          value={
            evaluation.training_rows.toLocaleString()
          }
        />

        <Metric
          label="Final test rows"
          value={
            evaluation.test_rows.toLocaleString()
          }
        />

        <Metric
          label="Model features"
          value={
            evaluation.feature_count.toLocaleString()
          }
        />
      </div>

      <div
        className="
          mt-6
          grid
          gap-3
          border-t
          border-[var(--color-border)]
          pt-5
          md:grid-cols-2
        "
      >
        <div
          className="
            flex
            gap-3
            rounded-[5px]
            bg-[var(--color-surface-sunken)]
            p-4
          "
        >
          <Database
            size={15}
            strokeWidth={1.7}
            className="
              mt-0.5
              shrink-0
              text-[var(--color-accent)]
            "
          />

          <div>
            <p
              className="
                text-[9px]
                font-semibold
                uppercase
                tracking-[0.08em]
                text-[var(--color-text-tertiary)]
              "
            >
              Training period
            </p>

            <p
              className="
                mt-1
                text-[11px]
                leading-5
                text-[var(--color-text-secondary)]
              "
            >
              {formatDatasetTimestamp(
                evaluation.train_start,
              )}
              {' → '}
              {formatDatasetTimestamp(
                evaluation.train_end,
              )}
              {' UTC'}
            </p>
          </div>
        </div>

        <div
          className="
            flex
            gap-3
            rounded-[5px]
            bg-[var(--color-surface-sunken)]
            p-4
          "
        >
          <CalendarRange
            size={15}
            strokeWidth={1.7}
            className="
              mt-0.5
              shrink-0
              text-[var(--color-accent)]
            "
          />

          <div>
            <p
              className="
                text-[9px]
                font-semibold
                uppercase
                tracking-[0.08em]
                text-[var(--color-text-tertiary)]
              "
            >
              Final holdout period
            </p>

            <p
              className="
                mt-1
                text-[11px]
                leading-5
                text-[var(--color-text-secondary)]
              "
            >
              {formatDatasetTimestamp(
                evaluation.test_start,
              )}
              {' → '}
              {formatDatasetTimestamp(
                evaluation.test_end,
              )}
              {' UTC'}
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}