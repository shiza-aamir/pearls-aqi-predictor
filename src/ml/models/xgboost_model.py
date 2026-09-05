from xgboost import XGBRegressor


def create_xgboost_model(
    random_state: int = 42,
) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=1,
        random_state=random_state,
    )