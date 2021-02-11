from datetime import date

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

MLFLOW_ARTIFACT_ROOT = "/tmp/mlruns"
MIN_SAMPLE_OUTPUT = 35

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
    df["ds"] = pd.to_datetime(df["ds"])
    df['weekday'] = df['ds'].apply(lambda x: x.weekday())
    df['year'] = df.ds.dt.year
    df['month'] = df.ds.dt.month
    df['day'] = df.ds.dt.day

    X = df.set_index("ds").drop(columns=["y"], errors="ignore")

    return X


def create_line_plot(X_test, y_test, yhat_test):
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(x=X_test.index, y=y_test, label='y', ax=ax)
    sns.lineplot(x=X_test.index, y=yhat_test, label='yhat', ax=ax)
    ax.legend(loc='upper left')
    ax.set(title='y vs yhat', ylabel='')

    return fig, ax


def run_experiment():
    with mlflow.start_run(run_name=f"traffic_prediction_{current_date}"):
        # Load and prepare training and validation data
        df = pd.read_csv(
            'https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv',
        )
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

        # Save plot to MLFlow
        fig, ax = create_line_plot(X_test, y_test, yhat_test)
        fig.savefig(f"{MLFLOW_ARTIFACT_ROOT}/line_plot.png")
        plt.close()
        mlflow.log_artifact(f"{MLFLOW_ARTIFACT_ROOT}/line_plot.png")

        # Save a sample of raw data as an artifact
        sample_data = X_test.sample(min(MIN_SAMPLE_OUTPUT, len(X_test)))
        sample_data.to_csv(f"{MLFLOW_ARTIFACT_ROOT}/sample_data.csv")
        mlflow.log_artifact(f"{MLFLOW_ARTIFACT_ROOT}/sample_data.csv")


if __name__ == "__main__":
    run_experiment()
