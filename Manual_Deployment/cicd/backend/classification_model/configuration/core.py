from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

# Project Directories
# path associated to the folder where tox file is located
# this method need to import folder "classification_model"
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent.parent

CLASSIFICATION_MODEL_DIR = PACKAGE_ROOT / "classification_model"
DATASET_DIR = PACKAGE_ROOT / "classification_model" / "datasets"
CONFIG_FILE_PATH = PACKAGE_ROOT / "classification_model" / "config.yml"
TRAINED_MODEL_DIR = (
    PACKAGE_ROOT / "classification_model" / "save_trained_model_elements"
)

DATA_PREPARATION_PIPELINE_DIR = TRAINED_MODEL_DIR / "data_preparation_pipeline"
FEATURES_ENGINEERING_PIPELINE_DIR = TRAINED_MODEL_DIR / "features_engineering_pipeline"
TRAINING_ESTIMATOR_PIPELINE_DIR = TRAINED_MODEL_DIR / "training_estimator_pipeline"


# Configuration will be used like application in the model, like pipeline, ...
class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    data_preparation_pipeline_save_file: str
    feature_engineering_pipeline_save_file: str
    classification_model_pipeline_save_file: str
    data_preparation_pipeline_dir: str
    features_engineering_pipeline_dir: str
    training_estimator_pipeline_dir: str


# Configuration used in for model building like features, targets, random seed, ...
class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features_after_dp: List[str]
    train_features: List[str]
    initial_features: List[str]
    initial_variables: List[str]
    test_size: float
    random_state: int
    C: float
    n_lots: int
    search_missing_value_list: List[str]
    categorical_variables_first_split: List[str]
    categorical_variables_extract_title: List[str]
    categorical_variables: List[str]
    categorical_variables_with_na_missing: List[str]
    categorical_variables_with_na_frequent: List[str]
    others_variables: List[str]
    rare_variables: List[str]
    drop_variables: List[str]
    numerical_variables_casted: List[str]
    numerical_variables: List[str]
    numerical_variables_with_na: List[str]
    discrete_variables: List[str]
    continuous_variables: List[str]
    numerical_quantile_variables: List[str]
    binarize_variables: List[str]
    outliers_variables: List[str]


class MasterConfig(BaseModel):
    """Master config object."""

    app_config: AppConfig
    mod_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"config.yml not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """check the type of variable witch will be pass in the function"""

    # if not isinstance(cfg_path, Union[Path, None]):
    #    raise ValueError('variable passed in cfg_path should be a Path or None type')

    if cfg_path is None:
        pass  # print("cfg_path is None")
    elif not isinstance(cfg_path, Path):
        raise ValueError("variable passed in cfg_path should be a Path or None type")

    # """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            # parsed_config = yaml.load(conf_file, Loader=yamlordereddictloader.Loader)
            return parsed_config
    raise OSError(f"Did not find config.yml file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> MasterConfig:
    """check the type of variable witch will be pass in the function"""

    if parsed_config is None:
        pass  # print("parsed_config is None")
    elif not isinstance(parsed_config, YAML):
        raise ValueError(
            "variable passed in parsed_config should be a YAML or None type"
        )

    # """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = MasterConfig(
        app_config=AppConfig(**parsed_config.data),
        mod_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
