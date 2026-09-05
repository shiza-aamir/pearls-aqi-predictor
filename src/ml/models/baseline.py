import pandas as pd


class PersistenceBaseline:
    @staticmethod
    def predict(
        df: pd.DataFrame,
    ) -> pd.Series:
        if "aqi_current" not in df.columns:
            raise ValueError(
                "aqi_current column is required "
                "for the persistence baseline."
            )

        return df["aqi_current"].copy()