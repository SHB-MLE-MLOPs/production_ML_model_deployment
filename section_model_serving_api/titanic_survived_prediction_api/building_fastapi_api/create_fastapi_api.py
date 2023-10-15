from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from pathlib import Path
import sys
sys.path.append(
    str(Path().absolute().parent))  # this command beginning in the folder of file prepare_fastapi_api.py everywhere you launch it, this path is in the folder section-production-model-package

from building_fastapi_api import prepare_fastapi_api
from building_fastapi_api import config_fastapi_api

# create a default APIRouter named root_router (it's concerning information, you will have when you open browser and navigate to right url indicates in the console (like localhost:8001 for example)),
# the goal of router is to define and organize API routes, endpoints, and request handlers in a structured way
# #root_router = APIRouter()


# to define one API GET endpoint within root_router.
# concerning information, you will have or get when you open browser and navigate to right url indicates in the console (like localhost:8001 for example),
# and also at the last GET endpoint in the API
# html language is used to write this
# #@root_router.get("/")
# #def for_html_response(request: Request) -> Any:
# #    """Basic HTML response."""
# #    body = (
# #        "<html>"
# #        "<body style='padding: 10px;'>"
# #        "<h1>Welcome to the API</h1>"
# #        "<div>"
# #        "Check the docs: <a href='/docs'>here</a>"
# #        "</div>"
# #        "</body>"
# #        "</html>"
# #    )

# #    return HTMLResponse(content=body)


# ===== CREATE THE ROOT API-ROOTER =====
# create a default APIRouter named root_router (it's concerning information, you will have when you open browser and navigate to right url indicates in the console (like localhost:8001 for example)),
# the goal of router is to define and organize API routes, endpoints, and request handlers in a structured way.
# but it's already created in the module prepare_fastapi_api with the code bellow :
# == root_router = APIRouter() ==


# ===== DEFINE ONE GET ENDPOINT OF THE ROOT API-ROOTER =====
# to define one API GET endpoint within root_router.
# concerning information, you will have or get when you open browser and navigate to right url indicates in the console (like localhost:8001 for example),
# and also at the last GET endpoint of the API
# html language is used to write this.
# but it's already created in the module prepare_fastapi_api with the code bellow :
# == @root_router.get("/")
# == def for_html_response(request: Request) -> Any:


# ===== CREATE THE API-ROOTER OF API =====
# create a default APIRouter named api_router for our API interface,
# the goal of router is to define and organize API routes, endpoints, and request handlers in a structured way.
# but it's already created in the module prepare_fastapi_api with the code bellow :
# == api_router = APIRouter() ==


# ===== DEFINE ONE GET ENDPOINT OF THE API API-ROOTER =====
# to define one API GET endpoint within api_router.
# concerning information, you will have or get in the first endpoint when you open the API,
# but it's already created in the module prepare_fastapi_api with the code bellow :
# == @api_router.get("/settings", response_model=schemas.SettingsGetEndpoint, status_code=200) ==
# == def for_api_settings() -> dict: ==


# ===== DEFINE ONE POST ENDPOINT OF THE API API-ROOTER =====
# to define one API POST endpoint within api_router.
# concerning information, that you can provide or post or give on API interface,
# but it's already created in the module prepare_fastapi_api with the code bellow :
# == @api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200) ==
# == async def for_prediction(input_data: schemas.MultipleHouseDataInputs) -> Any: ==


# ===== CREATE THE LOGGING AREA OF API =====
# to create the api setting's or the logging area of api before configuration of this logging area,
# but it's already created in the module config_fastapi_api with the code bellow :
# == settings = Settings() ==

# api_logging_area = config_fastapi_api.settings


# ===== CONFIGURE THE LOGGING AREA OF API =====
# configure the logging area of api (for customer users) with function "setup_api_logging(config: Settings)"
# setup logging as early as possible
config_fastapi_api.setup_api_logging(config=config_fastapi_api.settings)

# config_fastapi_api.setup_api_logging(config=api_logging_area)


# ===== CREATE THE API INTERFACE =====
# create api interface with FastAPI Web Frameworks
api = FastAPI(
    title=config_fastapi_api.settings.PROJECT_NAME, openapi_url=f"{config_fastapi_api.settings.ALL_ROUTE_PREFIX}/openapi.json"
)

# api = FastAPI(
#     title=config_fastapi_api.settings.PROJECT_NAME, openapi_url=f"{api_logging_area.ALL_ROUTE_PREFIX}/openapi.json"
# )


# ===== INCLUDE ROOT AND API API-ROOTER (WITH ALL ENDPOINTS) CREATED PREVIOUSLY IN THE API INTERFACE =====
# include api_router (witch will be visible in our api interface) and root_router in the api using api.include_router(),
# and for api_router including, we specify a URL prefix for all routes defined within the api_router.
api.include_router(prepare_fastapi_api.api_router, prefix=config_fastapi_api.settings.ALL_ROUTE_PREFIX)
api.include_router(prepare_fastapi_api.root_router)

# api.include_router(prepare_fastapi_api.api_router, prefix=api_logging_area.ALL_ROUTE_PREFIX)
# api.include_router(prepare_fastapi_api.root_router)


# ===== ADD MIDDLEWARE TO API =====
# Add middleware components to the FastAPI API and Set all CORS enabled origins
# to configure CORS (Cross-Origin Resource Sharing) middleware for the FastAPI API
if config_fastapi_api.settings.BACKEND_CORS_ORIGINS:
    api.add_middleware(   # to add middleware components to the FastAPI API. Middleware functions are called for each incoming HTTP request, allowing you to perform various operations before the request is handled by your endpoint.
        CORSMiddleware,  # to control which domains are allowed to make requests to the FastAPI API
        allow_origins=[str(origin) for origin in config_fastapi_api.settings.BACKEND_CORS_ORIGINS],  # to specify the list of origins (domains) that are allowed to access the FastAPI API
        allow_credentials=True,  # boolean to indicate whether credentials (e.g., cookies, HTTP authentication) should be included in cross-origin requests. Setting it to "True" allows credentials to be sent, while setting it to "False" prevents them.
        allow_methods=["*"],  # to specify the HTTP methods that are allowed for cross-origin requests. Setting it to ["*"] means that any HTTP method is allowed. You can restrict it to specific methods, such as ["GET", "POST"], if needed.
        allow_headers=["*"],  # to specify the HTTP headers that are allowed for cross-origin requests. Setting it to ["*"] means that any headers are allowed. You can restrict it to specific headers, such as ["Accept", "Authorization", "User-Agent", ...] if needed.
    )

# if api_logging_area.BACKEND_CORS_ORIGINS:
#     api.add_middleware(   # to add middleware components to the FastAPI API. Middleware functions are called for each incoming HTTP request, allowing you to perform various operations before the request is handled by your endpoint.
#         CORSMiddleware,  # to control which domains are allowed to make requests to the FastAPI API
#         allow_origins=[str(origin) for origin in config_fastapi_api.settings.BACKEND_CORS_ORIGINS],  # to specify the list of origins (domains) that are allowed to access the FastAPI API
#         allow_credentials=True,  # boolean to indicate whether credentials (e.g., cookies, HTTP authentication) should be included in cross-origin requests. Setting it to "True" allows credentials to be sent, while setting it to "False" prevents them.
#         allow_methods=["*"],  # to specify the HTTP methods that are allowed for cross-origin requests. Setting it to ["*"] means that any HTTP method is allowed. You can restrict it to specific methods, such as ["GET", "POST"], if needed.
#         allow_headers=["*"],  # to specify the HTTP headers that are allowed for cross-origin requests. Setting it to ["*"] means that any headers are allowed. You can restrict it to specific headers, such as ["Accept", "Authorization", "User-Agent", ...] if needed.
#     )


# ===== RUN THE API CREATED =====
if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn  # import the Web Server

    uvicorn.run(api, host="localhost", port=8002, log_level="debug")
