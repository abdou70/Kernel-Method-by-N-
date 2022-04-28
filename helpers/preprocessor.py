import pandas as pd
import numpy as np
import os
import re

CSV_PATH = os.path.join("..", "datasets", "raw", "index.csv")


def load_csv(csv_path: str = CSV_PATH,  sep: str =","):
    """
    Read a comma-separated values (csv) file into DataFrame."
    :param csv_path: str
        File path.
    :return:
        None
    """
    return pd.read_csv(csv_path, sep)

def split(data,percent):
    
    np.random.seed(1)
    
    perm=np.random.permutation(data.index)
    train=int(len(data)*percent)
    
    X_train=data.iloc[perm[:train],:-1]
    X_test = data.iloc[perm[train:],:-1]
    
    y_train = data.iloc[perm[:train],-1]
    y_test =data.iloc[perm[train:],-1]
    
    return X_train , X_test , y_train,y_test
    
