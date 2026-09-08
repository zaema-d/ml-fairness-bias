import numpy as np
from aif360.datasets import GermanDataset, StandardDataset
from ucimlrepo import fetch_ucirepo
from folktables import ACSDataSource, ACSEmployment
import pandas as pd


def load_german():
    return GermanDataset()


def load_taiwan():
    taiwan = fetch_ucirepo(id=350)
    df = taiwan.data.features.copy()

    # The raw UCI file uses generic names like X1, X2, ..., X23.
    # Rename them to meaningful column names so the dataset is easier to read
    # and easier to interpret later in the fairness pipeline.
    df = df.rename(columns={
        'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE',
        'X5': 'AGE', 'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3',
        'X9': 'PAY_4', 'X10': 'PAY_5', 'X11': 'PAY_6',
        'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 'X14': 'BILL_AMT3',
        'X15': 'BILL_AMT4', 'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6',
        'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2', 'X20': 'PAY_AMT3',
        'X21': 'PAY_AMT4', 'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6',
    })
    # Add the target variable.
    # 0 = no default, 1 = default.
    df['default'] = np.asarray(taiwan.data.targets).astype(int).ravel()

    
    # Original UCI codes use 1 = male, 2 = female.
    # Convert to the 0/1 protected-attribute format used by AIF360:
    # 1 = privileged (male), 0 = unprivileged (female).
   
    df['SEX'] = df['SEX'].map({1: 1, 2: 0})
    return StandardDataset(
        df=df,
        label_name='default',
        favorable_classes=[0],        # 0 = did not default (favorable)
        protected_attribute_names=['SEX'],
        privileged_classes=[[1]]      # 1 = male (privileged)
    )

    
def load_folktables(state="CA", year="2018"):
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True)
    features, label, group = ACSEmployment.df_to_pandas(acs_data)

    df = features.copy()
    # The label is returned as a separate array.
    # We attach it to the dataframe so the data is in one table for training.
    df['employed'] = np.asarray(label).astype(int).ravel()

    # UCI-style coding: 1 = male, 2 = female.
    # Convert to the same 1/0 convention used across the project:
    # 1 = privileged, 0 = unprivileged.
    df['SEX'] = df['SEX'].map({1: 1, 2: 0})

    return StandardDataset(
        df=df,
        label_name='employed',
        favorable_classes=[1],        # 1 = employed (favorable)
        protected_attribute_names=['SEX'],
        privileged_classes=[[1]]      # 1 = male (privileged)
    )