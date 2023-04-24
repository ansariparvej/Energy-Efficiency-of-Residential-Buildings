from energy_efficiency.logger import logging
from energy_efficiency.pipeline.training_pipeline import TrainPipeline
from energy_efficiency.utils.main_utils import read_yaml_file
from energy_efficiency.constant.training_pipeline import SAVED_MODEL_DIR
from fastapi import FastAPI
from energy_efficiency.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from energy_efficiency.ml.model.estimator import ModelResolver
from energy_efficiency.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

env_file_path = os.path.join(os.getcwd(), "env.yaml")


def set_env_variable(env_file_path):

    if os.getenv('MONGO_DB_URL', None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL'] = env_config['MONGO_DB_URL']


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training is successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/predict")
async def predict_route():
    try:
        df = pd.read_csv('predict.csv')
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available.")

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred = model.predict(df)
        print(y_pred)
        return Response("Prediction is successfully done. check your IDE console for prediction result. ")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)


if __name__ == "__main__":
    # main()
    env_file_path = "Your Environment File Path"
    app_run(app, host=APP_HOST, port=APP_PORT)
