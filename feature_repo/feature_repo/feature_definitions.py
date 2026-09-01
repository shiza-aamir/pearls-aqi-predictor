from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
)
from feast.types import Float64


project = Project(
    name="pearls_aqi_predictor",
    description="Feature store for the Pearls AQI Predictor",
)


city = Entity(
    name="city",
    join_keys=["city_id"],
    description="City identifier used for AQI forecasting",
)


aqi_source = FileSource(
    name="aqi_model_features_source",
    path="data/model_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)


MODEL_FEATURE_FIELDS = [
    Field(name="latitude", dtype=Float64),
    Field(name="longitude", dtype=Float64),
    Field(name="temperature", dtype=Float64),
    Field(name="humidity", dtype=Float64),
    Field(name="precipitation", dtype=Float64),
    Field(name="wind_speed", dtype=Float64),
    Field(name="pressure", dtype=Float64),
    Field(name="pm2_5", dtype=Float64),
    Field(name="pm10", dtype=Float64),
    Field(name="carbon_monoxide", dtype=Float64),
    Field(name="nitrogen_dioxide", dtype=Float64),
    Field(name="sulphur_dioxide", dtype=Float64),
    Field(name="ozone", dtype=Float64),
    Field(name="hour_sin", dtype=Float64),
    Field(name="hour_cos", dtype=Float64),
    Field(name="day_of_week_sin", dtype=Float64),
    Field(name="day_of_week_cos", dtype=Float64),
    Field(name="month_sin", dtype=Float64),
    Field(name="month_cos", dtype=Float64),
    Field(name="is_weekend", dtype=Float64),
    Field(name="wind_direction_sin", dtype=Float64),
    Field(name="wind_direction_cos", dtype=Float64),
    Field(name="temp_humidity_interaction", dtype=Float64),
    Field(name="stagnation_index", dtype=Float64),
    Field(name="aqi_lag_1h", dtype=Float64),
    Field(name="aqi_lag_3h", dtype=Float64),
    Field(name="aqi_lag_6h", dtype=Float64),
    Field(name="aqi_lag_12h", dtype=Float64),
    Field(name="aqi_lag_24h", dtype=Float64),
    Field(name="aqi_lag_48h", dtype=Float64),
    Field(name="aqi_lag_72h", dtype=Float64),
    Field(name="pm2_5_lag_1h", dtype=Float64),
    Field(name="pm2_5_lag_3h", dtype=Float64),
    Field(name="pm2_5_lag_6h", dtype=Float64),
    Field(name="pm2_5_lag_24h", dtype=Float64),
    Field(name="pm10_lag_1h", dtype=Float64),
    Field(name="pm10_lag_3h", dtype=Float64),
    Field(name="pm10_lag_6h", dtype=Float64),
    Field(name="pm10_lag_24h", dtype=Float64),
    Field(name="aqi_rolling_mean_3h", dtype=Float64),
    Field(name="aqi_rolling_mean_6h", dtype=Float64),
    Field(name="aqi_rolling_mean_12h", dtype=Float64),
    Field(name="aqi_rolling_mean_24h", dtype=Float64),
    Field(name="aqi_rolling_std_3h", dtype=Float64),
    Field(name="aqi_rolling_std_6h", dtype=Float64),
    Field(name="aqi_rolling_std_12h", dtype=Float64),
    Field(name="aqi_rolling_std_24h", dtype=Float64),
    Field(name="pm2_5_rolling_mean_6h", dtype=Float64),
    Field(name="pm2_5_rolling_mean_24h", dtype=Float64),
    Field(name="pm10_rolling_mean_6h", dtype=Float64),
    Field(name="pm10_rolling_mean_24h", dtype=Float64),
    Field(name="aqi_change_1h", dtype=Float64),
    Field(name="aqi_change_3h", dtype=Float64),
    Field(name="aqi_change_24h", dtype=Float64),
    Field(name="pm2_5_change_1h", dtype=Float64),
    Field(name="pm10_change_1h", dtype=Float64),
]


aqi_features = FeatureView(
    name="aqi_features",
    entities=[city],
    ttl=timedelta(days=7),
    schema=MODEL_FEATURE_FIELDS,
    online=True,
    source=aqi_source,
    tags={
        "project": "pearls-aqi-predictor",
        "domain": "air-quality",
        "environment": "development",
        "feature_count": "56",
    },
    enable_validation=True,
)


aqi_feature_service = FeatureService(
    name="aqi_prediction_service",
    features=[aqi_features],
)