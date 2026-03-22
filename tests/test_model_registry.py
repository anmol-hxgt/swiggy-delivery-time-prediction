import pytest
import mlflow
from mlflow import MlflowClient
import dagshub

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

    assert len(latest_versions) > 0, f"No model found in {stage} stage"

    latest_version = latest_versions[0]

    # verify artifacts exist before loading
    run_id = latest_version.run_id
    artifacts = client.list_artifacts(run_id, "delivery_time_pred_model")
    assert len(artifacts) > 0, (
        f"No artifacts found for run {run_id}. "
        f"Re-run evaluate.py and register_model.py to upload fresh artifacts."
    )

    # load model
    model_uri = f"runs:/{run_id}/delivery_time_pred_model"
    model = mlflow.pyfunc.load_model(model_uri)

    assert model is not None, "Failed to load model from registry"

    print(
        f"Model '{model_name}' version {latest_version.version} "
        f"loaded successfully from {stage} stage"
    )