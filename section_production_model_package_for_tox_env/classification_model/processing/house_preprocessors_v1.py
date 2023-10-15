# -*- coding: utf-8 -*-
"""house_preprocessors.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_A8zOnQ9sY-x3aFuxHF75bV0kUFdcyl9
"""

# to handle datasets
import numpy as np
import pandas as pd

# base or mother object export from scikit-learn
# for getting and setting variables compatible to scikit-learn variables
from sklearn.base import BaseEstimator

# for fit_transform object compatible to scikit-learn
# you need to write fit and transform method in child object construction
from sklearn.base import TransformerMixin


# Replace all missing value in the data set by NAN
class ReplaceMissingValueByNanTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer
    def __init__(self, missing_values_list):
        if not isinstance(missing_values_list, list):
            raise ValueError('variable_list should be a list')

        self.missing_values_list = missing_values_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be a dataframe')

        # so that we do not over-write the original dataframe
        X = X.copy()

        for missing_value in self.missing_values_list:
            X = X.replace(missing_value, np.nan)

        return X


# Casting numerical variables as float

class CastingNumericalVariableAsFloat(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, casted_variable_list):
        if not isinstance(casted_variable_list, list):
            raise ValueError('variable_list should be a list')

        self.casted_variable_list = casted_variable_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be a dataframe')

        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.casted_variable_list:
            X[feature] = X[feature].astype('float')

        return X


# Retaining the first of cabin value if more than one

class RetainingFirstOfCabinTransform(BaseEstimator, TransformerMixin):
    def is_not_used(self):
        pass

    def __init__(self, variable_list):
        if not isinstance(variable_list, list):
            raise ValueError('variable_list should be a list')

        self.variable_list = variable_list

    def get_first_cabin(self, row: str) -> str:
        self.is_not_used()
        try:
            return row.split()[0]
        except:
            return np.nan

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be a dataframe')

        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.variable_list:
            X[feature] = X[feature].apply(self.get_first_cabin)

        return X


# Extraction title from the name of the variable

class ExtractionTitleFromTheNameTransform(BaseEstimator, TransformerMixin):
    def is_not_used(self):
        pass

    def __init__(self, variable_list):
        if not isinstance(variable_list, list):
            raise ValueError('variable_list should be a list')

        self.variable_list = variable_list

    def get_title(self, passenger: str) -> str:
        import re

        line = passenger
        self.is_not_used()
        if re.search('Mrs', line):
            return 'Mrs'
        elif re.search('Mr', line):
            return 'Mr'
        elif re.search('Miss', line):
            return 'Miss'
        elif re.search('Master', line):
            return 'Master'
        else:
            return 'Other'

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be a dataframe')

        # so that we do not over-write the original dataframe
        X = X.copy()

        X['title'] = X['name'].apply(self.get_title)
        # for feature in self.variable_list:
        #    X[feature] = X[feature].apply(self.get_title)

        return X


# Extraction of the letter and keep it only in the variable Cabin

class VariableCabinTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variable_list):
        if not isinstance(variable_list, list):
            raise ValueError('variable_list should be a list')

        self.variable_list = variable_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be a dataframe')

        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.variable_list:
            X[feature] = X[feature].str[0]

        return X


# Create new features that capture information about presence or absence of Outliers in data set

class OutliersFeatureCreation(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, outliers_num_vars_list):
        if not isinstance(outliers_num_vars_list, list):
            raise ValueError('outliers_num_vars should be a list')

        self.outliers_num_vars_list = outliers_num_vars_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be a dataframe')

        # so that we do not over-write the original dataframe
        X = X.copy()

        # capture the outliers and create the new feature
        for var in self.outliers_num_vars_list:
            # Identify outliers using IQR
            Q1 = np.percentile(X[var], 25)
            Q3 = np.percentile(X[var], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (X[var] < lower_bound) | (X[var] > upper_bound)

            # add outliers indicator for each column with outliers data
            X[var + '_outliers'] = np.where(outliers, 1, 0)

        return X


# Temporal elapsed time transformer

class TemporalVariableTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X


# for mapping categorical variable like quality, ... : when we have mapping dictionary

class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X


# for imputation of missing numerical variables (replaced by the mean or median or ...)

class MeanImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables):
        self.imputer_dict_ = None
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        # persist mean values in a dictionary
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature],
                              inplace=True)
        return X


# for encoding Rare labels (categorical variable)

class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Groups infrequent categories into a single string"""

    def __init__(self, variables, tol=0.05):
        self.encoder_dict_ = None
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.tol = tol
        self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts(normalize=True))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                X[feature], "Rare")

        return X


# for regression problem
# one way for encoding categorical variable to capture monotonic relationship between categorical variables and target

# this object will assign discrete values to the strings of the variables,
# so that the smaller value corresponds to the category that shows the smaller
# mean house sale price

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables):
        self.encoder_dict_ = None
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])["target"].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X
