import cudf
import numpy as np
import os
import pandas as pd
import requests

from cuml.preprocessing import OneHotEncoder, MinMaxScaler


def kdd99_clean_dataset(df):
    """
    Function to prepare the KDD99 dataset for preprocessing.
    
    It includes:
    - Dropping duplicates
    - Assigning column names
    - Removing one-valued (non-variable) columns
    - Removing the dot at the end of each output variable observation

    Args
    ----
        df (_pandas.Dataframe_): Pandas dataframe containing the data.

    Returns
    -------
        _pandas.Dataframe_: The original dataframe, free of duplicates, invariant features, and with named columns.
    """
    clean_df = df.copy()
    clean_df.drop_duplicates()
    
    # Obtain columns name
    url = "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.names"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        lines = content.split("\n")
        # First line contains only the name of attacks (output)
        # Each feature line is instead of the format 'col_name: type'
        attribute_lines = [line.strip() for line in lines if ":" in line] # Attacks line does not have ":"
        attribute_names = [line.split(":")[0].strip() for line in attribute_lines] + ["attack"]
    # Add names to columns
    clean_df.rename({k:v for (k, v) in zip(range(len(clean_df.columns)), attribute_names)}, axis=1, inplace=True)
    
    # If a column has only one value across all observations, it does not explain the response variable at all
    no_variability_cols = [col for col in clean_df.columns if clean_df[col].nunique() == 1]
    clean_df.drop(columns=no_variability_cols, inplace=True)
    
    # Finally, remove the dot at the end of each response value
    clean_df["attack"] = clean_df["attack"].apply(lambda x: x[:-1])
    
    return clean_df


def kdd99_cat_preprocessing(df):
    """
    Function to preprocess categorical data of the KDD99 dataset.
    
    A one-hot encoding method for the features is used, while a label encoding method is used for the output.

    Args:
        df (_pandas.Dataframe_): Pandas dataframe containing the cleaned data.

    Returns:
        _pandas.Dataframe_: The dataframe with one-hot encoded features and label encoded outputs.
    """
    cat_cols = [col for col in df.columns if (df[col].dtype not in ('int64', 'float64')) and (col != "attack")]
    # GPU processing from cuml library
    prep_df = cudf.from_pandas(df)

    ohe = OneHotEncoder(drop='first', sparse_output=False)
    encoded_array = ohe.fit_transform(prep_df[cat_cols])
    
    new_col_names = []
    for col, cats in zip(cat_cols, ohe.categories_):
        # Skip first category due to drop='first'
        for cat in cats[1:].to_pandas():
            new_col_names.append(f"{col}_{cat}")
    
    # Create cuDF DataFrame from encoded array
    encoded_df = cudf.DataFrame(encoded_array, columns=new_col_names)
    
    # Drop original categorical columns and concatenate encoded features
    prep_df = prep_df.drop(columns=cat_cols)
    prep_df = cudf.concat([prep_df, encoded_df], axis=1)
    
    # Reorder columns: move the specified column to the end
    prep_df = prep_df[[col for col in prep_df.columns if col != "attack"] + ["attack"]]
    
    # In response variable: Turn attacks in 1, normal in 0
    def encode_response(attack):
        if attack != "normal":
            return 1
        return 0

    prep_df["attack"] = prep_df["attack"].apply(encode_response)
    
    # Convert back to pandas DataFrame
    return prep_df.to_pandas()


def kdd99_num_preprocessing(df):
    """
    Function to preprocess numerical data of the KDD99 dataset.
    
    It expects all categorical features to be already encoded.
    
    It performs MinMaxScaling over the range of [0, 1].

    Args:
        df (_pandas.Dataframe_): Pandas dataframe containing the cleaned data, with all numerical values.

    Returns:
        _type_: _description_
    """
    mms = MinMaxScaler()
    
    X = df.drop(columns=['attack'])
    y = df['attack']

    
    # For some mystique reason, cuml's fit_transform method for some classes resets indices
    # We store the names in a variable so we can manually add them back
    feature_columns = X.columns.tolist()
    
    mms.fit(X)
    
    X = mms.transform(X)
    X.rename({k:v for (k, v) in zip(range(len(feature_columns)), feature_columns)}, axis=1, inplace=True)
    
    prep_df = pd.concat([X, y], axis=1)
    
    return prep_df


def preprocessing_pipeline(df):
    """
    The pipeline for the preprocessing steps of the KDD99 dataset.

    Args:
        df (_pandas.Dataframe_): Pandas dataframe containing the data.

    Returns:
        _pandas.Dataframe_: Preprocessed pandas dataframe, ready to use in various algorithms.
    """
    df = kdd99_clean_dataset(df)
    df = kdd99_cat_preprocessing(df)
    df = kdd99_num_preprocessing(df)
    return df


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("./data/")
    df = pd.read_csv("kddcup.data_10_percent_corrected", header=None)
    preprocessed_df = preprocessing_pipeline(df)
    print(preprocessed_df.head())
