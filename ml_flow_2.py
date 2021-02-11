from datetime import date

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import mlflow

mlflow_settings = dict(
    username="mlflow",
    password="mlflow",
    host="127.0.0.1",
    port=5000,
)

mlflow.set_tracking_uri(
    "http://{username}:{password}@{host}:{port}".format(**mlflow_settings)
)

current_date = date.today()
experiment_id = mlflow.set_experiment("Web Traffic Forecast")


def prepare_data(df):
    df["date"] = pd.to_datetime(df["date"])
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day

    X = df.set_index("date").drop(columns=["y"], errors="ignore")

    return X


def run_experiment():
    with mlflow.start_run(run_name=f"traffic_prediction_{current_date}"):
        # Load and prepare training and validation data
        df = pd.read_csv("dog_wiki_views.csv")
        X = prepare_data(df)
        Y = df["y"]

        # Run validation on data
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.15, shuffle=False
        )
        model = XGBRegressor()
        model.fit(X_train, y_train)
        yhat_test = model.predict(X_test)

        # Compute MSE metric
        mse = mean_squared_error(y_test, yhat_test)
        mlflow.log_metric("MSE", mse)

        # Track features
        mlflow.log_param("Features", X.columns.tolist())
        mlflow.log_param("Date", current_date)


if __name__ == "__main__":
    run_experiment()
