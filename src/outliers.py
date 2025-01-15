import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest



def detect_outliers_iqr(df, column):
  """
    Detects outliers in a column using the Interquartile Range (IQR) method.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame
    column : str
        Column name to check for outliers

    Returns:
    --------
    pandas DataFrame
        Rows identified as outliers
    """
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
  return outliers


def detect_outliers_zscores(df,column,threshold=3):
  """
    Detects outliers in a column using the Z-Score method.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame
    column : str
        Column name to check for outliers
    threshold : float, default=3
        Z-Score threshold for identifying outliers

    Returns:
    --------
    pandas DataFrame
        Rows identified as outliers
    """
  df['z_score'] = zscore(df[column])
  outliers = df[df['z_score'].abs() > threshold]
  df.drop(columns=['z_score'], inplace= True)
  return outliers


def detect_outliers_isolation_forest(df, column, contamination=0.5):
  """
    Detects outliers using Isolation Forest.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame
    columns : list
        List of columns to consider for outlier detection
    contamination : float, default=0.05
        Proportion of outliers in the data

    Returns:
    --------
    pandas DataFrame
        Rows identified as outliers
  """
  iso = IsolationForest(contamination=contamination, random_state=42)
  df['outlier'] = iso.fit_predict(df[[column]])
  outliers = df[df['outlier'] == -1]
  df.drop(columns=['outlier'], inplace=True)
  return outliers


def handle_outliers(df, column, method='remove', lower_bound=None):
  """
    Handles outliers in a column by removing or capping them.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame
    column : str
        Column name to handle outliers
    method : str, default='remove'
        Method to handle outliers: 'remove', 'cap'
    lower_bound : float, optional
        Lower bound for capping (used when method='cap')
    upper_bound : float, optional
        Upper bound for capping (used when method='cap')

    Returns:
    --------
    pandas DataFrame
        DataFrame with outliers handled
    """
  df_copy = df.copy()

  if method == 'remove':
    Q1 = df_copy[column].quantile(0.25)
    Q3 = df_copy[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_copy = df_copy[(df_copy[column] >= lower_bound ) & (df_copy[column] <= upper_bound)]

  elif method == 'cap':
    if lower_bound is None or upper_bound is None:
      Q1 = df_copy[column].quantile(0.25)
      Q3 = df_copy[column].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      df_copy[column] = df_copy[column].clip(lower = lower_bound, upper = upper_bound)

  else:
    raise ValueError("Unsupported method. Choose 'remove' or 'cap'. ")

  return df_copy

def remove_outliers(df, column, threshold):
  """
    Removes rows from the DataFrame where the specified column's values exceed the given threshold.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame from which outliers are to be removed.
    column : str
        The name of the column to check for outliers.
    threshold : float or int
        The maximum acceptable value in the specified column.
        Rows with values greater than this threshold will be removed.

    Returns:
    --------
    pandas DataFrame
        A DataFrame with outliers removed based on the specified threshold.

    Notes:
    ------
    - This function assumes that outliers are defined as values strictly greater than the given threshold.
    - Rows where the column's value is less than or equal to the threshold are retained.
  """

  df_copy = df.copy()

  df_copy = df_copy[df_copy[column] <= threshold]
  return df_copy