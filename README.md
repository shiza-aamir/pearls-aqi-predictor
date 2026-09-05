# PEARLS AQI Production Monitoring State

This branch stores durable post-deployment forecast monitoring state.

The forecast ledger is updated automatically by the hourly GitHub Actions
production pipeline. It is intentionally separated from the application
source-code branch so monitoring updates do not trigger application
redeployments.

Do not use this branch for model selection or training.
