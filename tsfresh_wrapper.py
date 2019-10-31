import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# TSFRESH
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import EfficientFCParameters, MinimalFCParameters


class TSFreshTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column_id, column_sort, column_value, extraction_settings):
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_value = column_value
        self.extraction_settings = extraction_settings

    def fit(self, train_x, train_y=None, **fit_params):
        return self

    def transform(self, X_train, y_train=None, **fit_params):
        X_features = extract_features(
            X_train,
            column_id=self.column_id,
            column_sort=self.column_sort,
            column_value=self.column_value,
            default_fc_parameters=self.extraction_settings,
            disable_progressbar=True)

        impute(X_features)
        return X_features

    def get_params(self, **kwargs):
        return {'column_id': self.column_id,
                'column_sort': self.column_sort,
                'column_value': self.column_value,
                'extraction_settings': self.extraction_settings}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)


def generate_ts_features(df, X_train, X_test, date_col):
    def merge(df, features, date_col):
        return df.merge(features, how='left', left_on='building_id__meter__key', right_on='id', right_index=False,
                        suffixes=('', f'_{str(date_col)}'))

    features = TSFreshTransformer('building_id__meter__key', date_col, 'meter_reading', MinimalFCParameters()) \
        .fit_transform(df, axis=0)

    X_train = merge(X_train, features, date_col)
    X_test = merge(X_test, features, date_col)

    return X_train, X_test


def generate_ts(X_train_val, y_train_val, X_test, date_col):
    assert X_train_val.shape[0] > X_test.shape[0]
    assert X_train_val.shape[0] == y_train_val.shape[0]

    X_train_val_func = X_train_val.copy()
    X_test_func = X_test.copy()

    X = pd.concat([X_train_val.reset_index(), y_train_val.reset_index()], axis=1)

    X_train_val_func, X_test_func = generate_ts_features(X, X_train_val_func, X_test_func, date_col)

    X_train_val_func = X_train_val_func.fillna(0)
    X_test_func = X_test_func.fillna(0)

    return X_train_val_func, X_test_func
