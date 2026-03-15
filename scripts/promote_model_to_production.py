import mlflow
import dagshub
from mlflow.tracking import MlflowClient

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

# stage where model currently exists
STAGING_STAGE = "Staging"

# stage we want to move to
PRODUCTION_STAGE = "Production"


def promote_model():

    client = MlflowClient()

    # get latest model version from staging
    versions = client.get_latest_versions(
        name=MODEL_NAME,
        stages=[STAGING_STAGE]
    )

    if not versions:
        raise Exception("No model found in Staging stage")

    latest_version = versions[0].version

    print(f"Promoting model version {latest_version} to Production")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage=PRODUCTION_STAGE,
        archive_existing_versions=True
    )

    print(f"Model version {latest_version} promoted to Production successfully")


if __name__ == "__main__":
    promote_model()