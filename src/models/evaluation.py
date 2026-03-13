import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score


# initialize dagshub
dagshub.init(
    repo_owner="anmol-hxgt",
    repo_name="swiggy-delivery-time-prediction",
    mlflow=True
)

# set MLflow tracking server
mlflow.set_tracking_uri(
    "https://dagshub.com/anmol-hxgt/swiggy-delivery-time-prediction.mlflow"
)

# set experiment
mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

# logger setup
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

handler.setFormatter(formatter)
logger.addHandler(handler)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error("The file to load does not exist")


def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"

    model_path = root_path / "models" / "model.joblib"

    # load data
    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")

    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")

    # split features and target
    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test, y_test = make_X_and_y(test_data, TARGET)
    logger.info("Data split completed")

    # load model
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("Predictions completed")

    # evaluation metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    logger.info("Metrics calculated")

    # cross validation
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    mean_cv_score = -(cv_scores.mean())

    logger.info("Cross validation complete")

    # MLflow logging
    with mlflow.start_run():

        mlflow.set_tag("model", "Food Delivery Time Regressor")

        # log parameters
        mlflow.log_params(model.get_params())

        # log metrics
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("mean_cv_score", mean_cv_score)

        mlflow.log_metrics({
            f"CV_{num}": score for num, score in enumerate(-cv_scores)
        })

        # dataset logging
        train_dataset = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_dataset = mlflow.data.from_pandas(test_data, targets=TARGET)

        mlflow.log_input(dataset=train_dataset, context="training")
        mlflow.log_input(dataset=test_dataset, context="validation")

        # model signature
        model_signature = mlflow.models.infer_signature(
            model_input=X_train.sample(20, random_state=42),
            model_output=model.predict(X_train.sample(20, random_state=42))
        )

        # log model
        mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=model_signature
)

        # log additional artifacts
        mlflow.log_artifact(root_path / "models" / "stacking_regressor.joblib")
        mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")
        mlflow.log_artifact(root_path / "models" / "preprocessor.joblib")

        logger.info("MLflow logging complete")