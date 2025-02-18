import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class UserPredictor:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("forest", RandomForestClassifier(
                n_estimators=120,
                random_state=66
            ))
        ])
        self.feature_columns = None
        
    def combine(self, users, logs, y=None):
        
        df_users = users
        df_logs = logs
        df_y = y

        variables = [df_users, df_logs]
        if y is not None:
            variables.append(df_y)
        dataframes = all(isinstance(var, pd.DataFrame) for var in variables)

        if not dataframes:
            raise ValueError("not dataframes")

        df = pd.merge(df_users, df_logs, on='user_id', how='left')

        if y is not None:
            df = pd.merge(df, df_y, on='user_id', how='left')
            df["y"] = df["y"].astype(int)

        agg_dict = {
            "seconds": "sum",
            "past_purchase_amt": "sum",
            "age": "first"
        }

        if y is not None:
            agg_dict["y"] = "first"

        if "badge" in df.columns:
            agg_dict["badge"] = "first"
        if "url" in df.columns:
            agg_dict["url"] = "first"

        df = df.groupby("user_id", as_index=False).agg(agg_dict)

        if "badge" in df.columns:
            df = pd.get_dummies(df, columns=["badge"], drop_first=True)
        if "url" in df.columns:
            df = pd.get_dummies(df, columns=["url"], drop_first=True)

        return df


            
    def fit(self, users, logs, y):
        df = self.combine(users, logs, y)
        X = df.drop(columns=["y"])
        y = df["y"]
        
        if "user_id" in X.columns:
            X = X.drop("user_id", axis=1)
        if "date" in X.columns:
            X = X.drop("date", axis=1)
        if "names" in X.columns:
            X = X.drop("names", axis=1)
            
        self.feature_columns = X.columns
        
        self.model.fit(X, y)

        
    def predict(self, users, logs):
        df = self.combine(users, logs, y=None)
        
        if "y" in df.columns:
            df = df.drop("y", axis=1)
        if "user_id" in df.columns:
            df = df.drop("user_id", axis=1)
        if "date" in df.columns:
            df = df.drop("date", axis=1)
        if "names" in df.columns:
            df = df.drop("names", axis=1)
        
        df = df[self.feature_columns]
        
        prediction = self.model.predict(df)
        return prediction
