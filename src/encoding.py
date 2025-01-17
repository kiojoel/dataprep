import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_data(df, method='label', columns=None):
    """
    Encodes categorical variables in the DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing the data to be encoded.
    method : str, default='label'
        The encoding method to use:
        - 'label': Uses Label Encoding (integer encoding for each category).
        - 'onehot': Uses One-Hot Encoding (creates binary columns for each category).
    columns : list or None, default=None
        List of columns to encode. If None, all object or category columns are encoded.

    Returns:
    --------
    pandas DataFrame
        A DataFrame with encoded categorical variables.

    Notes:
    ------
    - For one-hot encoding, new columns are added for each category, and the original column is removed.
    - For label encoding, the original column is replaced with encoded integers.
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['object', 'category']).columns

    if method == 'label':
        le = LabelEncoder()
        for col in columns:
            df_copy[col] = le.fit_transform(df_copy[col])

    elif method == 'onehot':
        ohe = OneHotEncoder(sparse_output=False)
        for col in columns:
            encoded = ohe.fit_transform(df[[col]])
            encoded_columns = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([col]), index=df_copy.index)
            df_copy = pd.concat([df_copy, encoded_columns], axis=1)
            df_copy = df_copy.drop(columns=[col])


        """ for col in columns:
            encoded = pd.get_dummies(df_copy[col], prefix=col)
            df_copy = pd.concat([df_copy, encoded], axis=1)
            df_copy.drop(col, axis=1, inplace=True) """
    else:
        raise ValueError("Unsupported encoding method. Choose from: 'label', 'onehot'.")

    return df_copy
