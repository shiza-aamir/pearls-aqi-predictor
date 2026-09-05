# Pearls AQI Predictor

Pearls AQI Predictor is an end-to-end machine learning system for forecasting Air Quality Index (AQI) across major Pakistani cities at 24, 48, and 72-hour horizons.

It combines historical air-quality and meteorological data with live environmental observations, applies temporal feature engineering and walk-forward validation, serves production XGBoost models through FastAPI, and provides a React dashboard with forecasting, history, explainability, and live model monitoring.

## Live Application

**Frontend**  
https://pearls-aqi-predictor-two.vercel.app/

**Backend API**  
https://pearls-aqi-predictor-api.vercel.app/

**API Documentation**  
https://pearls-aqi-predictor-api.vercel.app/docs

## Key Features

- AQI forecasts for 24h, 48h, and 72h horizons
- Support for 9 major Pakistani cities
- Live weather and pollutant integration
- Historical AQI visualization
- XGBoost production models
- Temporal walk-forward validation
- Frozen final holdout evaluation
- SHAP-style XGBoost feature contribution explanations
- MLflow-managed model release workflow
- Feast-validated feature definitions
- Live forecast performance monitoring
- Automated CI and scheduled monitoring through GitHub Actions
- React and TypeScript production dashboard
- FastAPI backend

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

The system generates predictions for:

- 24 hours ahead
- 48 hours ahead
- 72 hours ahead

## Data Sources

The project uses environmental and meteorological observations from:

- Open-Meteo
- CAMS Global atmospheric data through Open-Meteo
- OpenWeather for live weather and pollutant observations
- AQICN as an optional external AQI validation source

AQI targets are derived using U.S. EPA particle-pollution breakpoint methodology.

## Machine Learning Pipeline

```text
Raw environmental data
        ↓
Data cleaning and alignment
        ↓
AQI target generation
        ↓
Temporal feature engineering
        ↓
56-feature ML dataset
        ↓
Walk-forward validation
        ↓
Classical and deep model benchmarking
        ↓
XGBoost champion selection
        ↓
Frozen final holdout evaluation
        ↓
Model release
        ↓
FastAPI inference
        ↓
React dashboard
        ↓
Live monitoring
Model Evaluation

Model selection was performed using temporal walk-forward validation.

The final holdout dataset remained isolated from model selection and was used only for final performance evaluation.

Horizon	MAE	RMSE	R²
24h	14.91	21.16	0.791
48h	19.35	26.89	0.654
72h	20.47	28.25	0.613

These values represent frozen final-holdout performance rather than training or model-selection scores.

Production Monitoring

Production forecasts are evaluated only after their target timestamps have matured.

The monitoring system tracks:

Mean Absolute Error
Root Mean Squared Error
AQI category accuracy
Adjacent-category accuracy
Evaluated forecast count
Next maturity timestamp

Live monitoring metrics are intentionally kept separate from frozen holdout metrics.

Explainability

Individual predictions are explained using native XGBoost feature contributions.

These contributions show which engineered features influence a forecast most strongly.

They describe model behaviour and should not be interpreted as causal effects.

MLOps

The project includes:

MLflow-managed production model releases
Feast-validated feature contracts
GitHub Actions continuous integration
Scheduled development model evaluation
Scheduled production monitoring
Durable forecast monitoring ledger
Versioned deployment metadata
Reproducible production artifacts
Technology Stack
Machine Learning
Python
Pandas
NumPy
Scikit-learn
XGBoost
MLOps
MLflow
Feast
GitHub Actions
Backend
FastAPI
Pydantic
Frontend
React
TypeScript
Vite
Deployment
Vercel
Project Structure
pearls-aqi-predictor/
├── api/                 FastAPI API layer
├── artifacts/           Evaluation and deployment metadata
├── feature_repo/        Feast feature definitions
├── frontend/            React/TypeScript dashboard
├── models/              Production model artifacts
├── pipelines/           Data and ML pipelines
├── scripts/             Training and evaluation utilities
├── src/                 Core ML and service logic
├── tests/               Automated tests
└── .github/workflows/   CI and scheduled automation
Local Setup
Backend

Clone the repository and create a virtual environment.

Install the project:

pip install -e ".[dev]"

Configure the required environment variables using .env.example.

Run the API:

uvicorn api.main:app --reload

The API will be available locally at:

http://localhost:8000

API documentation:

http://localhost:8000/docs
Frontend Setup

Navigate to the frontend directory:

cd frontend

Install dependencies:

npm install

Configure:

VITE_API_BASE_URL=http://localhost:8000/api/v1

Start the development server:

npm run dev
Testing

Backend:

ruff check src api pipelines tests
pytest

Frontend:

cd frontend
npm run lint
npm run build
Evaluation Protocol

The project follows a temporal evaluation strategy to reduce data leakage.

Model development uses walk-forward validation.

The final holdout set is isolated from model selection and is used only for final evaluation.

Scheduled development workflows do not automatically promote models based on final holdout performance.

Author

Shiza Aamir