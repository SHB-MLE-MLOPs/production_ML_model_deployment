import math

import numpy as np
import pandas as pd

from classification_model.configuration import core
from classification_model.new_prediction_with_model_trained import predict


def test_make_prediction(sample_input_data: pd.DataFrame):
    # check the type of variable witch will be pass in the function
    if not isinstance(sample_input_data, pd.DataFrame):
        raise ValueError(
            "variable passed in sample_input_data should be a DataFrame type"
        )
    # Given
    expected_first_prediction_value_1 = 0
    expected_first_prediction_value_2 = 1
    expected_no_predictions = sample_input_data.shape[0]  # 1449

    # When
    result = predict.make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    accuracyscore = result.get("AccuracyScore")
    rocaucscore = result.get("RocAucScore")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(
        predictions[0], expected_first_prediction_value_1, abs_tol=0.01
    ) or math.isclose(predictions[0], expected_first_prediction_value_2, abs_tol=0.01)

    # real VS prediction printing
    print()
    print()
    print("==================== THE MODEL PREDICTION ====================")
    print("prediction", predictions)
    print("==================== THE REAL RESPONSE ====================")
    print("real", sample_input_data[core.config.mod_config.target].values.tolist())
    print()

    # metrics for model evaluation printing
    print(
        "==================== METRICS BETWEEN REAL AND PREDICTION ===================="
    )
    print("test accuracy: {}".format(accuracyscore))
    print("test roc-auc: {}".format(rocaucscore))
