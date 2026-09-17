import pandas as pd
import numpy as np

files = {
    'Driver A (Huawei P20, UK) - S-S1': 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv',
    'Driver E (Huawei P20, UK) - S-Vta1a': 'data/raw/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vta (Driver E)/Vta01a/S-Vta1a.csv',
    'Driver F (Moto G7, France) - S-T1': 'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset/S-T1.csv',
    'Driver F (Moto G7, France) - S-T10': 'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset/S-T10.csv',
    'Driver G (Huawei P20, Nigeria) - S-I': 'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset/S-I.csv',
    'Driver H (Blackberry Priv, UK) - S-A1': 'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset/S-A1.csv',
    'Vehicle CAN (Ford Fiesta, UK) - V-S1': 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S1.csv'
}

for label, fpath in files.items():
    print('='*70)
    print(f'LABEL: {label}')
    print(f'PATH: {fpath}')
    df = pd.read_csv(fpath, encoding='latin1', nrows=100)
    print(f'Shape (nrows=100): {df.shape}')
    print('Columns:')
    for idx, col in enumerate(df.columns):
        # sample value and type
        sample_val = df[col].iloc[0]
        col_type = df[col].dtype
        has_zeros = (df[col] == 0).all()
        has_nans = df[col].isna().all()
        print(f'  [{idx:02d}] {repr(col)} -> sample: {sample_val}, dtype: {col_type}, all_zero: {has_zeros}, all_nan: {has_nans}')
