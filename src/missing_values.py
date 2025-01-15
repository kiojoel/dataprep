import pandas as pd
def fill_missing_values(df, method='mean', columns = 'None'):
   '''
    handles missing values in the specified columns of the DataFrame using the selected method.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame containing missing values to be filled.
    method : str, default='mean'
        Method to use for filling missing values:
        - 'mean': Fills missing values with the column's mean.
        - 'median': Fills missing values with the column's median.
        - 'mode': Fills missing values with the column's most frequent value.
        - 'drop': removes missing rows.
    columns : list or None, default=None
        List of columns to apply the missing value filling. If None, applies to all columns in the DataFrame.

    Returns:
    --------
    pandas DataFrame
        DataFrame with missing values filled or removed in the specified columns.

    Notes:
    ------
    - Non-numeric columns will raise an error for 'mean' and 'median' methods.
    - Ensure the selected method is appropriate for the data type of the columns.
    '''
   df_copy = df.copy()

   if columns is None:
    columns = df_copy.columns
   for col in columns:
    if method == 'mean':
      df_copy[col] = df[col].fillna(df[col].mean())
    elif method == 'median':
      df_copy[col] = df[col].fillna(df[col].median())
    elif method == 'mode':
      df_copy[col] = df[col].fillna(df[col].mode()[0])
    else:
      raise ValueError("Unsupported method")

   return df_copy

def drop_missing_values(df, axis=0, threshold=None):
  """
    Drops rows or columns with missing values.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame with missing values.
    axis : int, default=0
        Axis along which to drop:
        - 0: drops rows with missing values
        - 1: drops columns with missing values
    threshold : int or None, default=None
        Minimum non-NA values required to keep a row/column.
        If None, all rows/columns with any missing values will be dropped.

    Returns:
    --------
    pandas DataFrame
        DataFrame with rows or columns dropped.
    """
  df_copy = df.copy()

  if threshold is None:
    df_copy = df_copy.dropna(axis=axis)
  else:
    df_copy = df_copy.dropna(axis=axis, thresh=threshold)

  return df_copy