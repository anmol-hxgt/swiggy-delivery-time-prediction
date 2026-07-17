import mlflow
import dagshub
from mlflow.tracking import MlflowClient

dagshub.init(
    repo_owner="anmol-hxgt",
    repo_name="swiggy-delivery-time-prediction",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/anmol-hxgt/swiggy-delivery-time-prediction.mlflow"
)

MODEL_NAME = "delivery_time_pred_model"
PRODUCTION_STAGE = "Production"


def promote_model():
    client = MlflowClient()

    # get ALL versions, pick the highest version number regardless of current stage
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = max(all_versions, key=lambda v: int(v.version)).version

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