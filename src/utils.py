import pandas as pd
import numpy as np

#Split values into 3 sets
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

#Remove labels of a dataframe
def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

#Execute a diagnose of the data
def data_diagnose(df):
    '''
    This function will execute a diagnose of the data.
    '''
    summary = {}

    # 1. null values
    nulls = df.isnull().sum()
    summary['Nulls'] = nulls[nulls > 0]

    # 2. Infinit values
    infs = df.isin([np.inf, -np.inf]).sum()
    summary['Infinit values'] = infs[infs > 0]

    # 3. Outliers (por IQR)
    outliers = {}
    for col in df.select_dtypes(include=[np.number]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue  # Evita columnas constantes
        mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        count = mask.sum()
        if count > 0:
            outliers[col] = count
    summary['Outliers (IQR)'] = pd.Series(outliers)

    # Mostrar resultados
    for category, serie in summary.items():
        print(f"\n{category}:\n")
        print(serie)

    return  summary
