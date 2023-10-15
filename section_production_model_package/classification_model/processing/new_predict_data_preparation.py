from turtle import home
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)
from pydantic import BaseModel, ValidationError

from pathlib import Path
import sys
sys.path.append(
    str(Path().absolute().parent.parent))  # this command beginning in the folder of file new_predict_data_preparation.py everywhere you launch it, this path is in the folder section-production-model-package

from classification_model.configuration import core
from classification_model.processing import data_manager
from classification_model.processing import house_preprocessors_v1 as h_pp


# ===== TRANSFORM RAW PREDICT DATA SET INTO DATA READY FOR FEATURES ENGINEERING =====
def apply_data_exploration_transform(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Apply data preparation pipelines to raw predict data."""
    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError('variable passed in input_data should be a DataFrame type')

    predict_raw_data = input_data.copy()
    print(predict_raw_data)
    # data preparation pipelines applying
    # predict_data_dp = pipe_dp.transform(predict_raw_data)
    predict_data_dp = predict_raw_data

    return predict_data_dp


# ===== FUNCTION TO REPLACE ALL MISSING VALUE BY NAN IN PREDICT DATA SET =====
def replace_missing_by_nan_inputs_serge(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError('variable passed in input_data should be a DataFrame type')

    predict_raw_data = input_data.copy()
    # replace interrogation marks by NaN values
    for var in core.config.mod_config.search_missing_value_list:
        predict_raw_data = predict_raw_data.replace(var, np.nan)

    return predict_raw_data


# ===== FUNCTION TO REPLACE ALL MISSING VALUE BY NAN IN PREDICT DATA SET =====
def replace_missing_by_nan_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError('variable passed in input_data should be a DataFrame type')

    predict_raw_data = input_data.copy()
    validated_data = h_pp.ReplaceMissingValueByNanTransform(
        missing_values_list=core.config.mod_config.search_missing_value_list).transform(predict_raw_data)

    return validated_data


# ===== FUNCTION TO DROP THE NEW ENTIRE COLUMNS IN PREDICT DATA SET WITCH IS NOT PRESENT IN TRAIN DATA SET =====
def drop_new_variable_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError('variable passed in input_data should be a DataFrame type')

    predict_raw_data = input_data.copy()
    first_validated_data = replace_missing_by_nan_inputs(input_data=predict_raw_data)
    new_vars = [var for var in first_validated_data.columns
                if var not in core.config.mod_config.initial_variables
                ]
    second_validated_data = first_validated_data
    if new_vars:
        print(
            "DROP NEW VARIABLES : These following variables in the predict data set are new and are not present in training data set")
        print(new_vars)
        print(
            "DROP NEW VARIABLES: These variables will be drop to ensure the reproducibility of the model built in the prediction")
        second_validated_data = first_validated_data.drop(new_vars, axis=1)
        print(
            "DROP NEW VARIABLES : Therefore, the following list of variable will keep for further operations leading to new data set prediction")
        print(second_validated_data.columns)

    return second_validated_data


# ===== FUNCTION TO DROP ROW IF THE VARIABLE CONTAINS NAN IN PREDICT DATA SET BUT NOT CONTAINED IN THE TRAINING DATA SET =====
# when nan is present in some variable and this variable don't contain nan in training data set,
# we drop the row witch contain this nan value in the new predict data set.
# This ensures the reproducibility of the model built
def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError('variable passed in input_data should be a DataFrame type')

    predict_raw_data = input_data.copy()
    first_validated_data = drop_new_variable_inputs(input_data=predict_raw_data)
    new_vars_with_na = [var for var in first_validated_data.columns
                        if var not in core.config.mod_config.categorical_variables_with_na_frequent
                        + core.config.mod_config.categorical_variables_with_na_missing
                        + core.config.mod_config.numerical_variables_with_na
                        + core.config.mod_config.drop_variables
                        and first_validated_data[var].isnull().sum() > 0
                        ]

    if new_vars_with_na:
        print(
            "DROP ROW : These following variables in the predict data set have NaN value but doesn't have it in the training data set")
        print(new_vars_with_na)
        print(
            "DROP ROW : We will remove the raw witch contains these NaN value from the new predict data to ensure the reproducibility of the model built")
        first_validated_data.dropna(subset=new_vars_with_na, inplace=True)

    second_validated_data = first_validated_data

    return second_validated_data


# ===== FUNCTION FOR VALIDATION OF PREDICT DATA SET =====
def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError('variable passed in input_data should be a DataFrame type')

    # convert syntax error field names (beginning with numbers)
    # relevant_data = input_data[core.config.mod_config.initial_features].copy()  # extract data of good variable in raw data set
    relevant_data = input_data.copy()  # for raw data
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    # to be ensured that the predictive data witch will give to the predictive pipeline does not contain NaN value, you need to perform the bellow block try-except
    # This is mandatory when the pipeline use for prediction contains only the predictive pipeline
    # In others words, when the pipeline use for prediction don't integrate the entire way use to build the model (data preparation, target transformation,
    # feature engineering, features selection, model prediction, ...), you need to perform the bellow block try-except

    # So, in our case here, our hole training pipeline considers the entire way to build the model : data preparation pipeline, features engineering pipeline and prediction pipeline,
    # This way to build the hole training pipeline guaranteed us that we won't have NaN value in the predict dataset witch needed by our predictive pipeline,
    # so we don't need here to perform the bellow block try-except

    try:
        # replace numpy nans so that pydantic can validate
        # create the configuration for valid predict data set
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        # if we get error, put it in json file for reader
        errors = error.json()
        print(
            "VALID FEATURE FOR PREDICTION : Something wrong when execute the try block inside the function validate_inputs in the file new_predict_data_preparation")

    return validated_data, errors


# Create configuration for valid predict data set, like core.py and config.yml
# features used here are features in the data set witch gives to the pipeline on training step
class TitanicDataInputSchema(BaseModel):
    pclass: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[float]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[float]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[float]
    body: Optional[float]
    homedest: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
