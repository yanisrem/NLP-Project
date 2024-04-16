import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with median of column.
        """
    def fit(self, X, y=None):
        """
        Fit the imputer on the dataset.

        Args:
            X (pandas.DataFrame): Input DataFrame.
            y (array-like, optional): Target values.

        Returns:
            DataFrameImputer: Fitted DataFrameImputer object.
        """
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O')
                               else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        """
        Transform the input DataFrame.

        Args:
            X (pandas.DataFrame): Input DataFrame.
            y (array-like, optional): Target values.

        Returns:
            DataFrame: Transformed DataFrame with missing values imputed.
        """
        return X.fillna(self.fill)
