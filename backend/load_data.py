# backend/load_data.py
import pandas as pd

df = pd.read_csv("../data/train.csv")
print("Rows:", len(df))
print(df.head())
