import json
from typing import Any

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse

from loguru import logger

# sometimes, we don't have classification_model folder in our project directory, but we can import module from classification_model folder,
# because we package our model built in library named tsp-classification-model,
# witch was installed in the project virtual environment. So with this manipulation,
# we need to write correctly the path of import like in the packaging model step,
# and it will be recognized automatically

from pathlib import Path
import sys
sys.path.append(
    str(Path().absolute().parent))  # this command beginning in the folder of file prepare_fastapi_api.py everywhere you launch it, this path is in the folder section-production-model-package

from classification_model import __version__ as model_version
from classification_model.new_prediction_with_model_trained import predict

from building_fastapi_api import __version__, schemas
from building_fastapi_api.schemas import settings_endpoint, predict_endpoint
from building_fastapi_api import config_fastapi_api

# create a default APIRouter named root_router (it's concerning information, you will have when you open browser and navigate to right url indicates in the console (like localhost:8001 for example)),
# the goal of router is to define and organize API routes, endpoints, and request handlers in a structured way
root_router = APIRouter()


# to define one API GET endpoint within root_router.
# concerning information, you will have or get when you open browser and navigate to right url indicates in the console (like localhost:8001 for example),
# and also at the last GET endpoint in the API
# html language is used to write this
@root_router.get("/")
def html_response(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


# create a default APIRouter named api_router (This will be associated with our API interface),
# the goal of router is to define and organize API routes, endpoints, and request handlers in a structured way
api_router = APIRouter()


# to define one API GET endpoint within api_router.
# concerning information that you can get or read from API interface.
# Here, we define settings concerning our ML model, API, ... in this GET endpoint
# @api_router.get("/settings", response_model=schemas.SettingsGetEndpoint, status_code=200)  # this work because "schemas" is imported as module due to the presence of file __init__.py in the folder schemas, and in the file __init__.py we already import the class "SettingsGetEndpoint" from settings_endpoint.py
@api_router.get("/settings", response_model=settings_endpoint.SettingsGetEndpoint, status_code=200)
def settings() -> dict:
    """Root Get"""
    # api_settings = schemas.SettingsGetEndpoint(
    #     name=config_fastapi_api.settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    # )
    api_settings = settings_endpoint.SettingsGetEndpoint(
        name=config_fastapi_api.settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return api_settings.dict()


# to define an API POST endpoint within api_router.
# concerning information that you can post or give on API interface
# This POST endpoint of API concerns adding of the data we want to predict
# @api_router.post("/prediction", response_model=schemas.PredictionResultsPostEndpoint, status_code=200)  # this work because "schemas" is imported as module due to the presence of file __init__.py in the folder schemas, and in the file __init__.py we already import the classes "MultipleTitanicDataInputs" and "PredictionResultsPostEndpoint" from predict_endpoint.py
@api_router.post("/prediction", response_model=predict_endpoint.PredictionResultsPostEndpoint, status_code=200)
async def prediction(input_data: predict_endpoint.MultipleTitanicDataInputs) -> Any:
    """Make titanic survived predictions with the TID classification model"""
    # Here, input_data has predict_endpoint.MultipleTitanicDataInputs type witch means List[new_predict_data_preparation.TitanicDataInputSchema] type.
    # So before it working very well, it required that you give an input_data witch respect type of each element in TitanicDataInputSchema

    # to return Python objects (dataframe here) as JSON format before sending them as responses from your FastAPI endpoints
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = predict.make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results
