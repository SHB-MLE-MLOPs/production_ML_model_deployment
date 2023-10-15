import logging
import sys
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


# library logging is used to storage or persist the traffic on the api
# is used for the observability of our api built, along with things like metrics and distributed tracing

# class to set the API logging level
class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO  # logging levels are type int


# class for API settings
class Settings(BaseSettings):
    # prefix for all routes defined within the APIRouter witch communicate with the interface of API
    ALL_ROUTE_PREFIX: str = "/api/titanic.survived.predict/v1"

    # Meta
    # add logging level to the settings of API
    logging: LoggingSettings = LoggingSettings()

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    # e.g: http://localhost,http://localhost:4200,http://localhost:3000
    # to set a list of origins that the API can allow or approve or accept requests from (HTTP or HTTPS request, because browser application).
    # this list is for middleware we will add to API
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # type: ignore
        "http://localhost:8000",  # type: ignore
        "https://localhost:3000",  # type: ignore
        "https://localhost:8000",  # type: ignore
    ]

    # set the API name
    PROJECT_NAME: str = "Titanic Survived Prediction API"

    # For example, if you have a Pydantic model with case_sensitive = True, and you define an attribute as my_attribute,
    # then when you parse data, my_attribute must exactly match the case of the incoming data.
    # So, "my_attribute" and "My_Attribute" would be treated as different attributes.
    # If case_sensitive were set to False, Pydantic would perform case-insensitive matching for attribute names.
    # In that case, "my_attribute" and "My_Attribute" would be treated as the same attribute.
    class Config:
        case_sensitive = True


# the bellow class is for the case we want to intercept standard logging messages toward your Loguru sinks
# See: https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging  # noqa
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


# function to configure the API logging area
def setup_api_logging(config: Settings) -> None:
    """Prepare custom logging for the API."""

    # to inspect and replace the list of original handlers associated with a Logger into
    # handlers comes from interception of standard logging messages (customers logging with appropriate level) toward Loguru sinks
    LOGGERS = ("uvicorn.asgi", "uvicorn.access")  # attribute Logger (or Web Server). you need to have "Logger" before use "logging"
    logging.getLogger().handlers = [InterceptHandler()]  # in the Logger associated to logging, replace original list of handlers by list of handlers comes from interception of standard logging messages (customers logging with appropriate level) toward Loguru sinks
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)  # replace Logger (associated to logging) by our Logger we want to use
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]  # replace the level of our Logger handler's by the appropriate level for customers logging

    # Change the default configuration of "logger" into new configuration,
    # by replacing default handlers by the new one (by changing the sink and level) --> from loguru documentation.
    # "sink" is for handling incoming log messages and proceed to their writing somewhere and somehow
    # "sink": sys.stderr means that "sink" will handling the standard error stream (log messages here) witch comes from the console or terminal where a Python script is being executed
    # "level" is for the level of handlers (witch will be the appropriate level for logging)
    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    )


# create the API setting's or the logging area of API
settings = Settings()
