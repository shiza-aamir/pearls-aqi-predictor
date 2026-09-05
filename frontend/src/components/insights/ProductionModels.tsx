import {
  Boxes,
  GitBranch,
} from 'lucide-react'

import type {
  ProductionModel,
} from '../../api/types'

interface ProductionModelsProps {
  models: ProductionModel[]
}

export function ProductionModels({
  models,
}: ProductionModelsProps) {
  return (
    <section
      aria-labelledby="production-models-heading"
    >
      <div
        className="
          mb-4
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
            id="production-models-heading"
            className="
              text-[14px]
              font-semibold
            "
          >
            Production models
          </h2>

          <p
            className="
              mt-1
              text-[11px]
              text-[var(--color-text-tertiary)]
            "
          >
            Production models selected through
            validation, managed in MLflow, and
            exported as versioned champion
            artifacts for production serving.
          </p>
        </div>

        <div
          className="
            flex
            items-center
            gap-2
            text-[10px]
            uppercase
            tracking-[0.07em]
            text-[var(--color-text-tertiary)]
          "
        >
          <Boxes
            size={13}
            strokeWidth={1.8}
          />
          MLflow-managed release
        </div>
      </div>

      <div
        className="
          grid
          gap-3
          lg:grid-cols-3
        "
      >
        {models.map((model) => (
          <article
            key={model.horizon_hours}
            className="
              rounded-[6px]
              border
              border-[var(--color-border)]
              bg-[var(--color-surface)]
              p-5
            "
          >
            <div
              className="
                flex
                items-start
                justify-between
                gap-4
              "
            >
              <div>
                <p
                  className="
                    text-[11px]
                    font-semibold
                    uppercase
                    tracking-[0.08em]
                    text-[var(--color-text-tertiary)]
                  "
                >
                  {model.horizon_hours}-hour
                </p>

                <p
                  className="
                    font-display
                    mt-4
                    text-[31px]
                    font-medium
                    leading-none
                  "
                >
                  {model.algorithm.toUpperCase()}
                </p>
              </div>

              <GitBranch
                size={18}
                strokeWidth={1.7}
                className="
                  text-[var(--color-accent)]
                "
              />
            </div>

            <div
              className="
                mt-6
                space-y-3
                border-t
                border-[var(--color-border)]
                pt-4
              "
            >
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
                  Registry model
                </p>

                <p
                  className="
                    font-mono
                    mt-1
                    text-[11px]
                  "
                >
                  {model.registry_name}
                </p>
              </div>

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
                  Alias
                </p>

                <span
                  className="
                    font-mono
                    mt-1
                    inline-block
                    rounded-[4px]
                    bg-[var(--color-accent-soft)]
                    px-2
                    py-1
                    text-[10px]
                    text-[var(--color-accent-strong)]
                  "
                >
                  {model.registry_alias}
                </span>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}