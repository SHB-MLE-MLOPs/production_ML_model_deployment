from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import (AddMissingIndicator, CategoricalImputer,
                                       MeanMedianImputer)
from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (Binarizer, QuantileTransformer,
                                   StandardScaler)

from classification_model.configuration import core
from classification_model.processing import house_preprocessors_v1 as h_pp

# set up the pipeline
# using Feature-engine open source Library for building transformers

# ===== BUILDING PIPELINES FOR DATA SET PREPARATION =====
titanic_pipe_DP = Pipeline(
    [
        # == SEARCHING MISSING VALUE IN DATA SET AND REPLACE THEM BY NAN  ====
        (
            "replace_mv_by_NAN",
            h_pp.ReplaceMissingValueByNanTransform(
                missing_values_list=core.config.mod_config.search_missing_value_list
            ),
        ),
        # == CASTED SOME NUMERICAL VARIABLE AS FLOAT  ====
        (
            "some_nv_casted",
            h_pp.CastingNumericalVariableAsFloat(
                casted_variable_list=core.config.mod_config.numerical_variables_casted
            ),
        ),
        # == RETAINING ONLY THE FIRST IF MORE THAN 1 PER PASSENGER ====
        (
            "retain_first",
            h_pp.RetainingFirstOfCabinTransform(
                variable_list=core.config.mod_config.categorical_variables_first_split
            ),
        ),
        # == EXTRACTING OF THE TITLE (Mr, Ms, etc) FROM THE VARIABLE 'name'  ====
        (
            "extract_title",
            h_pp.ExtractionTitleFromTheNameTransform(
                variable_list=core.config.mod_config.categorical_variables_extract_title
            ),
        ),
        # == DROP UNNECESSARY VARIABLES FROM DATA SET  ====
        (
            "drop_features",
            DropFeatures(features_to_drop=core.config.mod_config.drop_variables),
        ),
    ]
)


# ===== BUILDING PIPELINES FOR DATA ENGINEERING =====
# set up the pipeline
# using Feature-engine open source Library for building transformers
titanic_pipe_FE = Pipeline(
    [
        # == KEEP IN THE COLUMN 'cabin' ONLY THE FIRST STRING  ====
        (
            "extract_letter",
            h_pp.VariableCabinTransform(
                variable_list=core.config.mod_config.categorical_variables_first_split
            ),
        ),
        # ===== IMPUTATION = TREATMENT OF MISSING VALUES =====
        # add missing indicator to missing value in numerical variables
        (
            "missing_indicator",
            AddMissingIndicator(
                variables=core.config.mod_config.numerical_variables_with_na
            ),
        ),
        # impute missing value in numerical variables with the median
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=core.config.mod_config.numerical_variables_with_na,
            ),
        ),
        # impute missing value in categorical variables
        # witch have higher percentage of missing values with string 'missing'
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=core.config.mod_config.categorical_variables_with_na_missing,
            ),
        ),
        # impute missing value in categorical variables
        # witch have few percentage of missing values with frequent value
        (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=core.config.mod_config.categorical_variables_with_na_frequent,
            ),
        ),
        # ==== OUTLIERS NUMERICAL VARIABLE TRANSFORMATION =====
        # create a new feature for presence or absence of outliers
        (
            "outliers_feature_creation",
            h_pp.OutliersFeatureCreation(
                outliers_num_vars_list=core.config.mod_config.outliers_variables
            ),
        ),
        # ==== NUMERICAL VARIABLE TRANSFORMATION =====
        # for variable witch becomes unskewed after apply transformation method
        (
            "quantile",
            SklearnTransformerWrapper(
                transformer=QuantileTransformer(
                    random_state=core.config.mod_config.random_state
                ),
                variables=core.config.mod_config.numerical_quantile_variables,
            ),
        ),
        # for variable that remains skewed even apply transformation method
        (
            "binarizer",
            SklearnTransformerWrapper(
                transformer=Binarizer(threshold=0),
                variables=core.config.mod_config.binarize_variables,
            ),
        ),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.05,
                n_categories=1,
                variables=core.config.mod_config.rare_variables,
            ),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True, variables=core.config.mod_config.others_variables
            ),
        ),
    ]
)


# ===== BUILDING PIPELINES TO TRAIN MODEL AND MAKE PREDICTIONS  =====
# set up the pipeline

# Create the estimators
estimator = LogisticRegression(
    C=core.config.mod_config.C, random_state=core.config.mod_config.random_state
)

# set scaler for data
feature_scaler = StandardScaler()

# Make pipeline with estimators ==> pipe
# using Feature-engine open source Library for building transformers
titanic_pipe_TE = Pipeline(
    [
        # scale using standardization
        ("scaler", feature_scaler),
        # logistic regression (use C=0.0005 and random_state=0)
        ("Logisticregression", estimator),
    ]
)
