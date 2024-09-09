import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import available_if

class ColumnRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, rename_dict):
        self.rename_dict = rename_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.rename(columns=self.rename_dict)

class ValueCreator(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            operations=None, 
            feature_names_out="one-to-one",
    ):
        self.operations = operations
        self.feature_names_out = feature_names_out

    def fit(self, X, y=None):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        return self

    def transform(self, X):
        for col_name, condition in self.operations.items():
            if isinstance(X, pd.DataFrame):
                X[col_name] = X.eval(condition)
            else:
                X = self._apply_condition_np(X, col_name, condition)
        self.feature_names_out_ = X.columns if isinstance(X, pd.DataFrame) else None
        return X
    
    def _apply_condition_np(self, X, col_name, condition):
        df = pd.DataFrame(X)
        df[col_name] = df.eval(condition)
        return df.values
    
    @available_if(lambda self: self.feature_names_out is not None)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(self, transform='pandas'):
        if transform == 'pandas':
            self.transform = self._wrapped_transform_pandas
        return self
    
    def _wrapped_transform_pandas(self, X): 
        return pd.DataFrame(self.transform(X), columns=X.columns, index=X.index)

class ValueConverter(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        mapping_dict=None,
        default_value="others",
        feature_names_out="one-to-one",
    ):
        self.mapping_dict = mapping_dict
        self.default_value = default_value
        self.feature_names_out = feature_names_out
   
    def fit(self, X, y=None):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        if self.mapping_dict is not None: 
            if isinstance(X, pd.DataFrame):
                return X.applymap(self._map_value)
            return np.vectorize(self._map_value)(X)
        return X
    
    @available_if(lambda self: self.feature_names_out is not None)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)
        
    def _map_value(self, value):
        return self.mapping_dict.get(value, self.default_value)

    def set_output(self, transform='pandas'):
        if transform == 'pandas':
            self.transform = self._wrapped_transform_pandas
        return self
    
    def _wrapped_transform_pandas(self, X): 
        return pd.DataFrame(self.transform(X), columns=X.columns, index=X.index)

class ValueClassifier(TransformerMixin, BaseEstimator):
    def __init__(self, 
        conditions=None, 
        choices=None, 
        default="Unknown",
        feature_names_out="one-to-one",
        rename="status",
    ):
        self.conditions = conditions
        self.choices = choices
        self.default = default
        self.feature_names_out = feature_names_out
        self.rename = rename
   
    def fit(self, X, y=None):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        return self

    def transform(self, X):
        if self.conditions is not None and self.choices is not None: 
            if isinstance(X, pd.DataFrame):
                return self._transform_df(X)
            return self._transform_array(X)
        return X
    
    def _transform_df(self, X):
        X = X.copy()
        condition = [self._apply_condition(cond, X) for cond in self.conditions]
        X[self.rename] = np.select(condition, self.choices, default=self.default)
        self.feature_names_out_ = X.columns if isinstance(X, pd.DataFrame) else None
        return X
    
    def _transform_array(self, X):
        if X.ndim == 1:
            return np.array([self.default for val in X])
        elif X.ndim == 2:
            condition = [self._apply_condition(cond, pd.DataFrame(X)) for cond in self.conditions]
            return np.select(condition, self.choices, default=self.default)
        else:
            raise ValueError("Unsupported array shape.")

    def _apply_condition(self, condition, X):
        if isinstance(X, pd.DataFrame):
            return X.eval(condition)
        else:
            raise ValueError("Unsupported input type for condition application.")
    
    @available_if(lambda self: self.feature_names_out is not None)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(self, transform='pandas'):
        if transform == 'pandas':
            self.transform = self._wrapped_transform_pandas
        return self
    
    def _wrapped_transform_pandas(self, X): 
        return pd.DataFrame(self.transform(X), columns=X.columns, index=X.index)