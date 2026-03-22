import pytest
import mlflow
import dagshub
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error

import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "anmol-hxgt"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "41716882be228c494f83e28f51ea10efea8501ed"

# initialize dagshub
dagshub.init(
    repo_owner="anmol-hxgt",
    repo_name="swiggy-delivery-time-prediction",
    mlflow=True
)

# set mlflow tracking uri
mlflow.set_tracking_uri(
    "https://dagshub.com/anmol-hxgt/swiggy-delivery-time-prediction.mlflow"
)

# -----------------------------
# Load model from registry
# -----------------------------

MODEL_NAME = "delivery_time_pred_model"
STAGE = "Production"

# load from latest production run URI to avoid stale artifacts
from mlflow import MlflowClient
client = MlflowClient()
versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
assert len(versions) > 0, f"No model found in {STAGE} stage"
run_id = versions[0].run_id
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/delivery_time_pred_model")

# -----------------------------
# Load preprocessor and test data
# -----------------------------

root_path = Path(__file__).parent.parent
test_data_path = root_path / "data" / "processed" / "test_trans.csv"
preprocessor = joblib.load(root_path / "models" / "preprocessor.joblib")


@pytest.mark.parametrize(
    "threshold_error",
    [5]
)
def test_model_performance(threshold_error):

    df = pd.read_csv(test_data_path)

    X = df.drop(columns=["time_taken"])
    y = df["time_taken"]

    # preprocess
    preprocessed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    preprocessed_df = pd.DataFrame(preprocessed, columns=feature_names)
    preprocessed_df['vehicle_condition'] = preprocessed_df['vehicle_condition'].astype('int64')

    y_pred = model.predict(preprocessed_df)

    mean_error = mean_absolute_error(y, y_pred)

    print("Mean Absolute Error:", mean_error)

    assert mean_error <= threshold_error, (
        f"Model failed performance threshold. "
        f"Error = {mean_error}, Threshold = {threshold_error}"
    )

    print(" Model passed performance test")