# import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from pprint36 import pprint

from pathlib import Path
import sys
sys.path.append(
    str(Path().absolute().parent.parent))  # this command beginning in the folder of file train_pipeline.py everywhere you launch it, this path is in the folder section-production-model-package

from classification_model.processing import data_manager
from classification_model.configuration import core
from classification_model.pipelines_building import pipelines

# from classification_model.processing import new_predict_data_preparation


# sys.path.append(os.path.dirname(str(Path().absolute().parent / "processing"))) # to have directory (or parent) of path

# sys.path.append(str(Path().absolute().parent / "processing"))
# pprint(sys.path)
# import data_manager

# sys.path.append(str(Path().absolute().parent.parent)) # this path is in the folder section-production-model-package
# pprint(sys.path)
# from classification_model.processing import data_manager

# sys.path.append(str(Path().absolute().parent / "configuration"))
# pprint(sys.path)
# import core

# sys.path.append(str(Path().absolute().parent.parent)) # this path is in the folder section-production-model-package
# pprint(sys.path)
# from classification_model.configuration import core


def run_training() -> None:
    """Train the model."""

    # ========== DATA SET LOADING ==========
    # read training data
    data = data_manager.load_dataset(file_name=core.config.app_config.training_data_file)

    # print(data)
    # data0 = new_predict_data_preparation.replace_missing_by_nan_inputs(input_data=data)
    # print(data0)
    # data1 = new_predict_data_preparation.drop_new_variable_inputs(input_data=data)
    # print(data1)
    # data2 = new_predict_data_preparation.drop_na_inputs(input_data=data)

    # ========== TARGET AND FEATURES ASSIGNMENT ==========
    # assign features and target of problem
    data_features = data[core.config.mod_config.initial_features]
    data_target = data[core.config.mod_config.target]

    # ========== BUILDING OF DATA PREPARATION PIPELINE ==========
    # train DP_pipeline to make the preparation of data set
    pipe_dp = pipelines.titanic_pipe_DP
    data_DP = pipe_dp.fit_transform(data_features, data_target)

    # persist trained DP_pipeline
    data_manager.save_pipeline(pipeline_save_dir_path=core.DATA_PREPARATION_PIPELINE_DIR,
                               pipeline_save_file=core.config.app_config.data_preparation_pipeline_save_file,
                               pipeline_to_persist=pipe_dp)

    # check if there is no variables witch contain missing values
    check = data_manager.check_missing_value(data_DP)
    print('==========================================================================')
    print(check[0], check[1])
    print('==========================================================================')
    # check if data transformation is performed well
    print(data_DP)
    print('==========================================================================')

    # ========== DATA SET SEPARATION INTO TRAIN AND TEST SET ==========
    # Splitting of data set into train and test --> When data set
    X_train, X_test, y_train, y_test = train_test_split(
        # data_AE.drop(core.config.mod_config.target, axis=1),  # predictors
        # data_AE[core.config.mod_config.target],
        data_DP,  # predictors
        data_target,
        test_size=core.config.mod_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=core.config.mod_config.random_state,
    )

    # ========== TARGET DATA TRANSFORMATION ==========
    # Transformation of target --> build pipeline for this step
    # y_train = np.log(y_train)

    # ========== BUILDING OF FEATURE ENGINEERING PIPELINES ==========
    # train titanic_pipe_FE to make the feature engineering of X_train and X_test
    pipe_fe = pipelines.titanic_pipe_FE
    X_train_transformed = pipe_fe.fit_transform(X_train)
    X_test_transformed = pipe_fe.transform(X_test)

    # persist trained AE_pipeline
    data_manager.save_pipeline(pipeline_save_dir_path=core.FEATURES_ENGINEERING_PIPELINE_DIR,
                               pipeline_save_file=core.config.app_config.feature_engineering_pipeline_save_file,
                               pipeline_to_persist=pipe_fe)

    # check if feature engineering is performed well
    print(X_train_transformed.info())
    print('==========================================================================')
    print(X_test_transformed.columns)
    print('==========================================================================')

    # ========== PERFORM FEATURE SELECTION TO REDUCE DATA SET TO THE SELECTED FEATURES ==========
    # when we want to perform feature selection
    # X_train_transformed_selected = X_train_transformed[core.config.mod_config.selected_features]
    # X_test_transformed_selected = X_test_transformed[core.config.mod_config.selected_features]

    # when we don't want to perform feature selection
    X_train_transformed_selected = X_train_transformed
    X_test_transformed_selected = X_test_transformed

    # ========== TRAINING MODEL FOR MAKING PREDICTION ==========
    # train titanic_pipe_TE with Gridsearch method in order to have a model trained and make predictions
    pipe_te = pipelines.titanic_pipe_TE
    # Create hyperparameter dictionary for GridSearchCV, specifying the parameters to search for
    param_grid = {}

    # Specification of the splitting method and its parameters
    cv = KFold(n_splits=core.config.mod_config.n_lots, shuffle=True, random_state=core.config.mod_config.random_state)

    # Make GridSearch with param_grid
    grid_search = GridSearchCV(estimator=pipe_te, param_grid=param_grid, scoring='neg_mean_squared_error',
                               cv=cv, refit=True)

    # Train the model using GridSearchCV on the TRAIN set (X_train, y_train)
    grid_search.fit(X_train_transformed_selected, y_train)

    # Write the best model and its parameters
    titanic_pipe_best_model = grid_search.best_estimator_
    titanic_pipe_best_model_param = grid_search.best_params_

    # persist trained titanic_pipe_TE
    data_manager.save_pipeline(pipeline_save_dir_path=core.TRAINING_ESTIMATOR_PIPELINE_DIR,
                               pipeline_save_file=core.config.app_config.training_estimator_pipeline_dir,
                               pipeline_to_persist=titanic_pipe_best_model)

    # ========== MAKE PREDICTION WITH MODEL TRAINED ==========
    # make predictions for train set
    class_train = titanic_pipe_best_model.predict(X_train_transformed_selected)
    pred_train = titanic_pipe_best_model.predict_proba(X_train_transformed_selected)[:, 1]

    # ========== TARGET DATA INVERSE TRANSFORMATION ==========
    # Transformation of target --> build pipeline for this step
    # class_train = np.exp(class_train)

    # ========== EVALUATE MODEL PERFORMANCE ==========
    # determine mse and rmse
    print('train roc-auc: {}'.format(roc_auc_score(y_train, pred_train)))
    print('train accuracy: {}'.format(accuracy_score(y_train, class_train)))
    print()

    # make predictions for test set
    class_test = titanic_pipe_best_model.predict(X_test_transformed_selected)
    pred_test = titanic_pipe_best_model.predict_proba(X_test_transformed_selected)[:, 1]

    # determine mse and rmse
    print('test roc-auc: {}'.format(roc_auc_score(y_test, pred_test)))
    print('test accuracy: {}'.format(accuracy_score(y_test, class_test)))
    print()


if __name__ == "__main__":
    run_training()
