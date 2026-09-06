# Pearls AQI Predictor

Pearls AQI Predictor is an end-to-end machine learning system for forecasting the **Air Quality Index (AQI)** across major Pakistani cities at **24, 48, and 72-hour horizons**.

The system combines historical air-quality and meteorological data with live environmental observations, applies temporal feature engineering and walk-forward validation, serves production XGBoost models through FastAPI, and provides an interactive React dashboard for forecasting, historical analysis, explainability, and live model monitoring.

## Live Application

**Frontend:**  
https://pearls-aqi-predictor-two.vercel.app/

**Backend API:**  
https://pearls-aqi-predictor-api.vercel.app/

**API Documentation:**  
https://pearls-aqi-predictor-api.vercel.app/docs

## Key Features

- AQI forecasting at 24h, 48h, and 72h horizons
- Support for 9 major Pakistani cities
- Live weather and pollutant integration
- Historical AQI and pollutant visualization
- 56 engineered production features
- XGBoost production forecasting models
- Temporal walk-forward validation
- Protected final holdout evaluation
- XGBoost feature-contribution explanations
- MLflow-managed model release workflow
- Feast-validated feature definitions
- Hazardous AQI alerts
- Live forecast performance monitoring
- Automated CI with GitHub Actions
- Hourly production forecasting and monitoring
- Scheduled development-model evaluation
- React and TypeScript production dashboard
- FastAPI prediction service
- Serverless cloud deployment

## Supported Cities

- Faisalabad
- Islamabad
- Karachi
- Lahore
- Multan
- Peshawar
- Quetta
- Rahim Yar Khan
- Sialkot

## Forecast Horizons

The system generates AQI predictions for:

- **24 hours ahead**
- **48 hours ahead**
- **72 hours ahead**

## Data Sources

The project uses environmental and meteorological observations from:

- **Open-Meteo** — historical and recent meteorological data
- **CAMS Global atmospheric data via Open-Meteo** — historical pollutant information
- **OpenWeather** — live weather and pollutant observations
- **AQICN** — optional external AQI validation

AQI values and forecasting targets are derived using the **U.S. EPA particle-pollution breakpoint methodology**.

## Machine Learning Pipeline

```text
Raw Environmental Data
        |
        v
Data Cleaning and Alignment
        |
        v
AQI Target Generation
        |
        v
Temporal Feature Engineering
        |
        v
56-Feature ML Dataset
        |
        v
Walk-Forward Validation
        |
        v
Classical and Deep Model Benchmarking
        |
        v
XGBoost Champion Selection
        |
        v
Frozen Final Holdout Evaluation
        |
        v
Model Release
        |
        v
FastAPI Inference
        |
        v
React Dashboard
        |
        v
Live Production Monitoring
```

## Model Development

The project evaluated both classical machine-learning and deep-learning approaches, including:

- Persistence baseline
- Ridge Regression
- Random Forest
- XGBoost
- CNN
- GRU
- CNN-LSTM

Model selection was performed using **temporal walk-forward validation** rather than random train/test splitting to reduce the risk of time-series leakage.

XGBoost was selected as the production model based on its validation performance across the three forecast horizons.

## Model Evaluation

The final holdout dataset remained isolated from model selection and was used only for final performance evaluation.

| Forecast Horizon | MAE | RMSE | R² |
|---|---:|---:|---:|
| 24h | 14.91 | 21.16 | 0.791 |
| 48h | 19.35 | 26.89 | 0.654 |
| 72h | 20.47 | 28.25 | 0.613 |

These values represent **frozen final-holdout performance**, rather than training or model-selection scores.

## Production Monitoring

Production forecasts are evaluated only after their corresponding target timestamps have matured. A prediction generated for a future 24h, 48h, or 72h target therefore remains pending until the actual AQI for that target time becomes available.

The monitoring system tracks:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- AQI category accuracy
- Adjacent-category accuracy
- Evaluated forecast count
- Next forecast maturity timestamp

Live production metrics are intentionally kept separate from frozen holdout metrics so that small early production samples are not presented as overall model performance.

## Explainability

Individual forecasts are explained using native **XGBoost feature contributions**.

The explanation service identifies the engineered features that contribute most strongly to an individual prediction. These values describe model behaviour and should not be interpreted as causal relationships.

## MLOps and Automation

The project includes:

- MLflow-managed production model releases
- Feast-validated feature contracts
- GitHub Actions continuous integration
- Hourly production forecasting
- Scheduled development-model evaluation
- Durable forecast monitoring ledger
- Live evaluation of matured forecasts
- Versioned deployment metadata
- Reproducible production model artifacts

The daily development workflow evaluates candidate models but does **not automatically promote or deploy them based on final holdout performance**.

## Technology Stack

| Layer | Technologies |
|---|---|
| Machine Learning | Python, Pandas, NumPy, Scikit-learn, XGBoost |
| Deep Learning Experiments | CNN, GRU, CNN-LSTM |
| MLOps | MLflow, Feast, GitHub Actions |
| Backend | FastAPI, Pydantic |
| Frontend | React, TypeScript, Vite |
| Deployment | Vercel |
| Production Monitoring | Durable forecast ledger and scheduled evaluation |

## Project Structure

```text
pearls-aqi-predictor/
|
|-- api/                  FastAPI API layer
|-- artifacts/            Deployment metadata
|-- data/                 Development data and protected splits
|-- feature_repo/         Feast feature definitions
|-- frontend/             React/TypeScript dashboard
|-- models/               Production XGBoost artifacts
|-- pipelines/            Automated ML pipelines
|-- scripts/              Data, training and evaluation utilities
|-- src/                  Core ML and service logic
|-- tests/                Automated tests
|-- .github/workflows/    CI and scheduled automation
|-- pyproject.toml        Python project configuration
`-- README.md             Project documentation
```

## Local Setup

### Backend

Clone the repository:

```bash
git clone https://github.com/shiza-aamir/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

Create and activate a Python virtual environment, then install the project:

```bash
pip install -e ".[dev]"
```

Configure the required environment variables using:

```text
.env.example
```

Start the FastAPI development server:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger/OpenAPI documentation:

```text
http://localhost:8000/docs
```

### Frontend

Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Configure the frontend API base URL:

```text
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

Start the development server:

```bash
npm run dev
```

## Testing

### Backend

Run static checks:

```bash
ruff check src api pipelines tests
```

Run automated tests:

```bash
pytest
```

### Frontend

```bash
cd frontend
npm run lint
npm run build
```

## Evaluation Protocol

The project follows a strictly temporal evaluation strategy to reduce data leakage.

Model development uses **walk-forward validation**, while the final holdout period remains isolated from model selection and is used only for final evaluation.

Scheduled development workflows evaluate candidate models independently and do not automatically use final holdout performance for model promotion.

## Project Report

A detailed project report covering data acquisition, exploratory analysis, feature engineering, model experimentation, temporal validation, deployment, explainability, automation, and production monitoring is included in the repository:

`pearlsAQI-project_report-shiza_aamir.pdf`

## Author

**Shiza Aamir**

Pearls AQI Predictor — End-to-End Machine Learning Internship Project