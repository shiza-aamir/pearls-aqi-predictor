export interface City {
  name: string
  latitude: number
  longitude: number
}

export interface CitiesResponse {
  cities: City[]
}

export interface Alert {
  level: string
  severity: number
  message: string
}

export interface Pollutants {
  pm2_5: number
  pm10: number
}

export interface Weather {
  temperature_c: number
  humidity_percent: number
  wind_speed_ms: number
}

export interface CurrentAirQuality {
  aqi: number
  category: string
  alert: Alert
  pollutants: Pollutants
  weather: Weather
}

export interface ForecastModel {
  name: string
  alias: string
}

export interface FeatureContribution {
  feature: string
  display_name: string
  feature_value: number
  contribution: number
  direction: 'increase' | 'decrease'
}

export interface ForecastExplanation {
  base_value: number
  top_features: FeatureContribution[]
}

export interface ForecastItem {
  horizon_hours: number
  target_at: string
  aqi: number
  category: string
  alert: Alert
  model: ForecastModel
  explanation: ForecastExplanation
}

export interface ForecastMetadata {
  data_source: string
  feature_count: number
  feature_store: string
  model_registry: string
  history_rows: number
}

export interface ForecastResponse {
  city: string
  observed_at: string
  timezone: string
  current: CurrentAirQuality
  forecasts: ForecastItem[]
  metadata: ForecastMetadata
}

export interface HistoryStatistics {
  minimum: number
  maximum: number
  average: number
  standard_deviation: number
}

export interface HistoryObservation {
  timestamp: string
  aqi: number
  category: string

  pm2_5: number
  pm10: number
  ozone: number
  nitrogen_dioxide: number
  sulphur_dioxide: number
  carbon_monoxide: number

  temperature_c: number
  humidity_percent: number
  wind_speed_ms: number

  source: string
}

export interface HistoryResponse {
  city: string
  start_time: string
  end_time: string
  requested_hours: number
  available_hours: number
  statistics: HistoryStatistics
  observations: HistoryObservation[]
}

export interface HoldoutPerformance {
  horizon_hours: number
  rows: number
  mae: number
  rmse: number
  r2: number
  median_absolute_error: number
  within_10_aqi_pct: number
  within_20_aqi_pct: number
  within_30_aqi_pct: number
  category_accuracy_pct: number
  persistence_mae: number
  mae_improvement_percent: number
}

export interface LivePerformance {
  horizon_hours: number
  evaluated_forecasts: number
  mae: number | null
  rmse: number | null
  within_10_aqi_pct: number | null
  within_20_aqi_pct: number | null
  within_30_aqi_pct: number | null
  category_accuracy_pct: number | null
  adjacent_category_accuracy_pct: number | null
}

export interface PerformanceResponse {
  city: string
  holdout_evaluation_label: string
  holdout: HoldoutPerformance[]
  live_status: string
  live_evaluated_forecasts: number
  live: LivePerformance[]
}

export interface ProductionModel {
  horizon_hours: number
  algorithm: string
  registry_name: string
  registry_alias: string
}

export interface ModelEvaluation {
  evaluation_type: string
  selection_metric: string
  selection_frozen_before_test: boolean
  training_rows: number
  test_rows: number
  cities: number
  feature_count: number
  train_start: string
  train_end: string
  test_start: string
  test_end: string
}

export interface ModelsResponse {
  production_models: ProductionModel[]
  evaluated_candidates: string[]
  evaluation: ModelEvaluation
}