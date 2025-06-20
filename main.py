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

# Univariate Analysis
numerical_cols = ['age_years', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'map']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('numerical_features_histograms.png')
plt.show()

# Univariate Analysis 
categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']

plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(x=col, data=df)
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('categorical_features_countplots.png')
plt.show()

# Bivariate Analysis 
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols + ['cardio']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features with Cardio')
plt.savefig('correlation_matrix.png')
plt.show()

# Bivariate Analysis 
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x='cardio', y=col, data=df)
    plt.title(f'{col} vs. Cardio')
    plt.xlabel('Cardio (0: No, 1: Yes)')
    plt.ylabel(col)
plt.tight_layout()
plt.savefig('numerical_features_vs_cardio_boxplots.png')
plt.show()

plt.figure(figsize=(15, 12))
for i, col in enumerate(categorical_cols):
    if col != 'cardio': 
        plt.subplot(3, 3, i + 1)
        sns.countplot(x=col, hue='cardio', data=df, palette='viridis')
        plt.title(f'{col} vs. Cardio')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.legend(title='Cardio')
plt.tight_layout()
plt.savefig('categorical_features_vs_cardio_countplots.png')
plt.show()