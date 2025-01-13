from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def normalize_data(df, method='minmax', columns=None):
    '''
    Normalizes specified columns in the DataFrame using various methods.

    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame to normalize.
    method : str, default='minmax'
        Method to use for normalization:
        - 'minmax': scales to range [0,1]
        - 'standard': standardizes to mean=0, std=1
        - 'robust': scales using statistics that are robust to outliers
        - 'decimal': divides by maximum absolute value.
    columns : list or None, default=None
        List of columns to normalize. If None, normalizes all numeric columns.

    Returns:
    --------
    pandas DataFrame
        Normalized DataFrame.
    '''
    df_copy = df.copy()  # Create a copy of the DataFrame

    # If no columns specified, normalize all numeric columns
    if columns is None:
        columns = df_copy.select_dtypes(include=['int64', 'float64']).columns

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'decimal':
        scaler = None  # No external scaler required for decimal scaling
    else:
        raise ValueError("Unsupported normalization method. Choose from: 'minmax', 'standard', 'robust', 'decimal'.")

    # Apply scaling
    for col in columns:
        if method == 'decimal':
            max_abs = abs(df_copy[col]).max()
            if max_abs > 0:  # Avoid division by zero
                df_copy[col] = df_copy[col] / max_abs
        else:
            df_copy[col] = scaler.fit_transform(df_copy[[col]])

    return df_copy
