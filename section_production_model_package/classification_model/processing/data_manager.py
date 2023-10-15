import typing as t

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from pprint36 import pprint

import os
from pathlib import Path
import sys
sys.path.append(
    str(Path().absolute().parent.parent)) # this command beginning in the folder of file data_manager.py everywhere you launch it, this path is in the folder section-production-model-package
from classification_model.configuration import core
from classification_model import __version__ as _version

# sys.path.append(os.path.dirname(str(Path().absolute().parent / "configuration")))
# from configuration import core

# sys.path.append(str(Path().absolute().parent / "configuration"))
# import core

# pprint(sys.path)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    # check the type of variable witch will be pass in the function
    if not isinstance(file_name, str):
        raise ValueError('variable passed in file_name should be a string')
    # load and copy the data set
    dataset = pd.read_csv(Path(f"{core.DATASET_DIR}/{file_name}")).copy()
    # dataset = pd.read_excel(Path(f"{core.DATASET_DIR}/{file_name}")).copy()
    return dataset


def save_pipeline(*, pipeline_save_dir_path: Path, pipeline_save_file: str, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
        Saves the versioned model, and overwrites any previous
        saved models. This ensures that when the package is
        published, there is only one trained model that can be
        called, and we know exactly how it was built.
        """

    # check the type of variable witch will be pass in the function
    if not isinstance(pipeline_save_dir_path, Path):
        raise ValueError('variable passed in pipeline_save_dir_path should be a Path type')
    if not isinstance(pipeline_save_file, str):
        raise ValueError('variable passed in pipeline_save_file should be a string type')
    if not isinstance(pipeline_to_persist, Pipeline):
        raise ValueError('variable passed in pipeline_to_persist should be a Pipeline type')

    # Prepare versioned save file name
    save_file_name = f"{pipeline_save_file}{_version}.pkl"
    save_path = pipeline_save_dir_path / save_file_name

    remove_old_pipelines(directory=pipeline_save_dir_path, files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def remove_old_pipelines(*, directory: Path, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    # check the type of variable witch will be pass in the function
    if not isinstance(directory, Path):
        raise ValueError('variable passed in directory should be a Path (directory of folder) type')
    if not isinstance(files_to_keep, list):
        raise ValueError('variable passed in files_to_keep should be a list (of string) type')

    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in directory.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def load_pipeline(*, directory: Path, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    # check the type of variable witch will be pass in the function
    if not isinstance(directory, Path):
        raise ValueError('variable passed in directory should be a Path (directory of folder) type')
    if not isinstance(file_name, str):
        raise ValueError('variable passed in file_name should be a string type')

    file_path = directory / file_name
    trained_pipeline = joblib.load(filename=file_path)
    return trained_pipeline


def check_missing_value(dataframe: pd.DataFrame) -> list:
    # check the type of variable witch will be pass in the function
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('variable passed in dataframe should be a DataFrame type')

    # list of string we want to check the presence
    missing_values_list = ['N/A', 'na', 'NaN', 'none', 'null', '-', '#N/A', 'n/a', 'NA', 'n/a', 'n.a.', 'undefined',
                           'unknown', 'missing', '?', '...']
    # load and copy the data set
    data = dataframe.copy()
    # check the missing values in data set
    check0 = data.isin(missing_values_list).sum()
    check1 = [var for var in data.columns if data[var].isin(missing_values_list).sum() > 0]
    return check0, check1
