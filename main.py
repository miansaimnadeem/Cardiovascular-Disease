import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Current working directory:", os.getcwd())
try:
	df = pd.read_csv(r'F:\Universty\4th Semester\Intro to Ds\cardio_train.csv')
except FileNotFoundError:
	print("Error: The file 'cardio_train.csv' was not found. Please check the file path and ensure the file exists in the directory above.")
	exit()

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataFrame Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

df = df.drop('id', axis=1)

df['age_years'] = (df['age'] / 365.25).astype(int)

df['gender'] = df['gender'].map({1: 0, 2: 1})

df = df[df['ap_hi'] > 0] 
df = df[df['ap_lo'] > 0] 
df = df[df['ap_hi'] >= df['ap_lo']] 

df = df[df['ap_hi'] < 250] 
df = df[df['ap_lo'] < 200] 

df['bmi'] = df['weight'] / ((df['height'] / 100)**2)

df['map'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3

print("First 5 rows of the dataset after feature engineering and cleaning:")
print(df.head())

print("\nDataFrame Info after feature engineering and cleaning:")
print(df.info())

print("\nDescriptive Statistics after feature engineering and cleaning:")
print(df.describe())