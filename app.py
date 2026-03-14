import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "anmol-hxgt"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "41716882be228c494f83e28f51ea10efea8501ed"

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import mlflow
import joblib
from sklearn import set_config

from scripts.data_clean_utils import perform_data_cleaning

# output as pandas
set_config(transform_output="pandas")

# initialize dagshub
import dagshub

dagshub.init(
    repo_owner="anmol-hxgt",
    repo_name="swiggy-delivery-time-prediction",
    mlflow=True
)

# MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/anmol-hxgt/swiggy-delivery-time-prediction.mlflow"
)

# ----------------------------
# Input schema
# ----------------------------

class Data(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


# ----------------------------
# Load model from MLflow registry
# ----------------------------

MODEL_NAME = "delivery_time_pred_model"
STAGE = "Staging"
model_path = f"models:/{MODEL_NAME}/{STAGE}"

try:
    model = mlflow.pyfunc.load_model(model_path)
    print("Model loaded from MLflow registry")
except Exception as e:
    print(f"MLflow load failed: {e}")
    print("Falling back to local model...")
    model = joblib.load("models/model.joblib")


# ----------------------------
# Load preprocessor
# ----------------------------

preprocessor = joblib.load("models/preprocessor.joblib")


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Swiggy Delivery Time Prediction API running"}


@app.post("/predict")
def do_predictions(data: Data):

    pred_data = pd.DataFrame(
        {
            "ID": data.ID,
            "Delivery_person_ID": data.Delivery_person_ID,
            "Delivery_person_Age": data.Delivery_person_Age,
            "Delivery_person_Ratings": data.Delivery_person_Ratings,
            "Restaurant_latitude": data.Restaurant_latitude,
            "Restaurant_longitude": data.Restaurant_longitude,
            "Delivery_location_latitude": data.Delivery_location_latitude,
            "Delivery_location_longitude": data.Delivery_location_longitude,
            "Order_Date": data.Order_Date,
            "Time_Orderd": data.Time_Orderd,
            "Time_Order_picked": data.Time_Order_picked,
            "Weatherconditions": data.Weatherconditions,
            "Road_traffic_density": data.Road_traffic_density,
            "Vehicle_condition": data.Vehicle_condition,
            "Type_of_order": data.Type_of_order,
            "Type_of_vehicle": data.Type_of_vehicle,
            "multiple_deliveries": data.multiple_deliveries,
            "Festival": data.Festival,
            "City": data.City,
        },
        index=[0],
    )

    # step 1: clean raw input
    cleaned_data = perform_data_cleaning(pred_data)
    print("cleaned_data shape:", cleaned_data.shape)
    print("cleaned_data columns:", cleaned_data.columns.tolist())
    print("cleaned_data values:\n", cleaned_data)

    # step 2: preprocess
    preprocessed = preprocessor.transform(cleaned_data)
    feature_names = preprocessor.get_feature_names_out()
    preprocessed_df = pd.DataFrame(preprocessed, columns=feature_names)
    preprocessed_df['vehicle_condition'] = preprocessed_df['vehicle_condition'].astype('int64')


    # step 3: predict
    prediction = model.predict(preprocessed_df)[0]

    return {"delivery_time_prediction": float(prediction)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)