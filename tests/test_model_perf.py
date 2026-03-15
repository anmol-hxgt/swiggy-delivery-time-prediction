import pytest
import mlflow
import dagshub
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

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
STAGE = "Staging"

model_path = f"models:/{MODEL_NAME}/{STAGE}"

model = mlflow.pyfunc.load_model(model_path)

# -----------------------------
# Load test data
# -----------------------------

root_path = Path(__file__).parent.parent
test_data_path = root_path / "data" / "processed" / "test_trans.csv"


@pytest.mark.parametrize(
    "threshold_error",
    [5]
)
def test_model_performance(threshold_error):

    df = pd.read_csv(test_data_path)

    X = df.drop(columns=["time_taken"])
    y = df["time_taken"]

    y_pred = model.predict(X)

    mean_error = mean_absolute_error(y, y_pred)

    print("Mean Absolute Error:", mean_error)

    assert mean_error <= threshold_error, (
        f"Model failed performance threshold. "
        f"Error = {mean_error}, Threshold = {threshold_error}"
    )

    print("✅ Model passed performance test")