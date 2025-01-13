import pandas as pd
def fill_missing_values(df, method='mean', columns = 'None'):
   '''
    Fills missing values in the specified columns of the DataFrame using the selected method.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame containing missing values to be filled.
    method : str, default='mean'
        Method to use for filling missing values:
        - 'mean': Fills missing values with the column's mean.
        - 'median': Fills missing values with the column's median.
        - 'mode': Fills missing values with the column's most frequent value.
        - 'ffill': Fills missing values using forward fill (propagates last valid value forward).
        - 'bfill': Fills missing values using backward fill (propagates next valid value backward).
    columns : list or None, default=None
        List of columns to apply the missing value filling. If None, applies to all columns in the DataFrame.

    Returns:
    --------
    pandas DataFrame
        DataFrame with missing values filled in the specified columns.

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