#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_and_clean_data(filepath):
    """Load data and clean column formats."""
    df = pd.read_csv(filepath, encoding='latin1')

    # Clean up specific columns
    df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(float)
    df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)

    # Drop rows with any missing values
    df = df.dropna()

    return df

def prepare_data(df, label_name="Price_euros"):
    """Prepare data for modeling: clean, encode, scale, split, and save."""
    # Features & label
    X = df.drop(columns=[label_name])
    y = df[label_name]

    # Define which columns are which
    numerical_features = ['Inches', 'Ram', 'Weight']
    categorical_features = ['Company', 'Product', 'TypeName', 'ScreenResolution',
                            'Cpu', 'Memory', 'Gpu', 'OpSys']

    # Preprocessing pipelines
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

    # Fit and transform features
    X_processed = preprocessor.fit_transform(X)

    # Save the preprocessor after fitting
    joblib.dump(preprocessor, "pipeline.joblib")

    # Reconstruct feature names
    cat_encoded = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_encoded)

    # Convert to DataFrame and reattach target
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    X_df[label_name] = y.values

    # Split into train/val/test
    train, temp = train_test_split(X_df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Save CSVs
    train.to_csv("laptop-data-train.csv", index=False)
    val.to_csv("laptop-data-val.csv", index=False)
    test.to_csv("laptop-data-test.csv", index=False)

    print("Data Preprocessing Complete and Saved!")
    print(f"Train set:     {train.shape}")
    print(f"Validation set:{val.shape}")
    print(f"Test set:      {test.shape}")

def main():
    filepath = "laptop-data.csv"
    df = load_and_clean_data(filepath)
    prepare_data(df, label_name="Price_euros")

if __name__ == "__main__":
    main()
