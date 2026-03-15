import pytest
import mlflow
from mlflow import MlflowClient
import dagshub

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

# your model name
MODEL_NAME = "delivery_time_pred_model"


@pytest.mark.parametrize(
    "model_name, stage",
    [(MODEL_NAME, "Staging")]
)
def test_load_model_from_registry(model_name, stage):

    client = MlflowClient()

    # get latest model version in staging
    latest_versions = client.get_latest_versions(
        name=model_name,
        stages=[stage]
    )

    latest_version = latest_versions[0].version if latest_versions else None

    assert latest_version is not None, f"No model found in {stage} stage"

    # load model
    model_path = f"models:/{model_name}/{stage}"

    model = mlflow.pyfunc.load_model(model_path)

    assert model is not None, "Failed to load model from registry"

    print(
        f"Model '{model_name}' version {latest_version} "
        f"loaded successfully from {stage} stage"
    )