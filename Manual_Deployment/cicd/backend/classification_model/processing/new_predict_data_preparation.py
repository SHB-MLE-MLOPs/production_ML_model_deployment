from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.configuration import core
from classification_model.processing import house_preprocessors_v1 as h_pp


# ===== FUNCTION TO REPLACE ALL MISSING VALUE BY NAN IN PREDICT DATA SET =====
def replace_missing_by_nan_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("variable passed in input_data should be a DataFrame type")

    predict_raw_data = input_data.copy()
    validated_data = h_pp.ReplaceMissingValueByNanTransform(
        missing_values_list=core.config.mod_config.search_missing_value_list
    ).transform(predict_raw_data)

    return validated_data


# ===== FUNCTION TO DROP THE NEW ENTIRE COLUMNS IN PREDICT DATA SET, =====
# ===== COLUMNS WITCH IS NOT PRESENT IN TRAIN DATA SET =====
def drop_new_variable_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""

    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("variable passed in input_data should be a DataFrame type")

    predict_raw_data = input_data.copy()
    first_validated_data = replace_missing_by_nan_inputs(input_data=predict_raw_data)
    new_vars = [
        var
        for var in first_validated_data.columns
        if var not in core.config.mod_config.initial_variables
    ]
    second_validated_data = first_validated_data
    if new_vars:
        print(
            "DROP NEW VARIABLES : These following variables in the predict data set",
            "are new and are not present in training data set",
        )
        print(new_vars)
        print(
            "DROP NEW VARIABLES: These variables will be drop to ensure",
            "the reproducibility of the model built in the prediction",
        )
        second_validated_data = first_validated_data.drop(new_vars, axis=1)
        print(
            "DROP NEW VARIABLES : Therefore, the following list of variable will keep",
            "for further operations leading to new data set prediction",
        )
        print(second_validated_data.columns)

    return second_validated_data


# ===== FUNCTION TO DROP ROW IF THE VARIABLE CONTAINS NAN IN PREDICT DATA SET BUT, =====
# ===== NOT CONTAINED IN THE TRAINING DATA SET =====
# When nan is present in some variable and this variable didn't contain nan
# in training data set, we drop the row witch contain this nan value,
# In the new predict data set. This ensures the reproducibility of the model built


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""

    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("variable passed in input_data should be a DataFrame type")

    predict_raw_data = input_data.copy()
    first_validated_data = drop_new_variable_inputs(input_data=predict_raw_data)
    new_vars_with_na = [
        var
        for var in first_validated_data.columns
        if var
        not in core.config.mod_config.categorical_variables_with_na_frequent
        + core.config.mod_config.categorical_variables_with_na_missing
        + core.config.mod_config.numerical_variables_with_na
        + core.config.mod_config.drop_variables
        and first_validated_data[var].isnull().sum() > 0
    ]

    if new_vars_with_na:
        print(
            "DROP ROW : These following variables in the predict data set",
            "have NaN value but doesn't have it in the training data set",
        )
        print(new_vars_with_na)
        print(
            "DROP ROW : We will remove the raw witch contains these NaN value",
            "from the new predict data to ensure the reproducibility of the model built",
        )
        first_validated_data.dropna(subset=new_vars_with_na, inplace=True)

    second_validated_data = first_validated_data

    return second_validated_data


# ===== FUNCTION FOR VALIDATION OF PREDICT DATA SET =====
def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # check the type of variable witch will be pass in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("variable passed in input_data should be a DataFrame type")

    # Convert syntax error field names (beginning with numbers).
    # Extract data of good variable in raw data set.
    # relevant_data = input_data[core.config.mod_config.initial_features].copy()
    relevant_data = input_data.copy()  # for raw data
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    # To be ensured that the predictive data witch will give to the predictive pipeline,
    # does not contain NaN value, you need to perform the bellow block try-except.
    # Mandatory when pipeline use for prediction contains only the predictive pipeline
    # In others words, when the pipeline use for prediction don't integrate,
    # the entire way use to build the model (data preparation, target transformation,
    # feature engineering, features selection, model prediction, ...),
    # you need to perform the bellow block try-except

    # So, in our case here, our hole training pipeline considers the entire way to build
    # the model: data preparation, features engineering and prediction pipelines.
    # This way to build the hole training pipeline guaranteed us that we won't
    # have NaN value in the predict dataset witch needed by our predictive pipeline.
    # So we don't need here to perform the bellow block try-except

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
            "VALID FEATURE FOR PREDICTION : Something wrong when execute the try block",
            "inside validate_inputs function in the file new_predict_data_preparation",
        )

    return validated_data, errors


# Create configuration for valid predict data set, like core.py and config.yml.
# Features used here are in the data set witch gives to the pipeline on training step
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
