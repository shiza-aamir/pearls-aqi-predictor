from sklearn.ensemble import RandomForestRegressor


def create_random_forest_model(
    random_state: int = 42,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=1,
        random_state=random_state,
    )