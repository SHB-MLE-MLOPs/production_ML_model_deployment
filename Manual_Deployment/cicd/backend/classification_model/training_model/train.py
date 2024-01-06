from sklearn.model_selection import train_test_split

from classification_model.configuration import core
from classification_model.pipelines_building import pipelines
from classification_model.processing import data_manager


def run_training() -> None:
    """Train the model."""

    # ========== DATA SET LOADING ==========
    # read training data
    data = data_manager.load_dataset(
        file_name=core.config.app_config.training_data_file
    )

    # ========== TARGET AND FEATURES ASSIGNMENT ==========
    # assign features and target of problem
    data_features = data[core.config.mod_config.initial_features]
    data_target = data[core.config.mod_config.target]

    # ========== BUILDING OF DATA PREPARATION PIPELINE ==========
    # train DP_pipeline to make the preparation of data set
    pipe_dp = pipelines.titanic_pipe_DP
    data_DP = pipe_dp.fit_transform(data_features, data_target)

    # persist trained DP_pipeline
    data_manager.save_pipeline(
        pipeline_save_dir_path=core.DATA_PREPARATION_PIPELINE_DIR,
        pipeline_save_file=core.config.app_config.data_preparation_pipeline_save_file,
        pipeline_to_persist=pipe_dp,
    )

    # ========== DATA SET SEPARATION INTO TRAIN AND TEST SET ==========
    # Splitting of data set into train and test --> When data set
    X_train, X_test, y_train, y_test = train_test_split(
        data_DP,  # predictors
        data_target,  # target
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

    # persist trained AE_pipeline
    data_manager.save_pipeline(
        pipeline_save_dir_path=core.FEATURES_ENGINEERING_PIPELINE_DIR,
        pipeline_save_file=core.config.app_config.feature_engineering_pipeline_save_file,
        pipeline_to_persist=pipe_fe,
    )

    # ========== PERFORM FEATURE SELECTION TO REDUCE DATA SET TO ==========
    # ========== THE SELECTED FEATURES ==========
    # when we want to perform feature selection
    # X_train_transformed_selected = X_train_transformed[
    #   core.config.mod_config.selected_features
    # ]

    # when we don't want to perform feature selection
    X_train_transformed_selected = X_train_transformed

    # ========== TRAINING MODEL FOR MAKING PREDICTION ==========
    # Train titanic_pipe_TE to have a model trained and make predictions.
    pipe_te = pipelines.titanic_pipe_TE
    pipe_te.fit(X_train_transformed_selected, y_train)

    # persist trained titanic_pipe_TE
    data_manager.save_pipeline(
        pipeline_save_dir_path=core.TRAINING_ESTIMATOR_PIPELINE_DIR,
        pipeline_save_file=core.config.app_config.training_estimator_pipeline_dir,
        pipeline_to_persist=pipe_te,
    )


if __name__ == "__main__":
    run_training()
