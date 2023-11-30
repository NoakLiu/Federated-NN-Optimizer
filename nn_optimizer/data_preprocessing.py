import numpy as np
import pandas as pd


def check_missing_data(df):
    # Check for any missing data in the DataFrame
    check = list(df.isnull().sum())
    miss = False
    for i in check:
        if i == 1:
            miss = True
            break
    return miss


class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mu) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)
