
def fill_missing_values(df, method='mean', columns = 'None'):
  '''
  Fills missing values in the DataFrame

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