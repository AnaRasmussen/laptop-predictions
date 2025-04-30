#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


# ========================
# Data Loading + Preprocessing
# ========================

def preprocess_for_xgboost(X_train, X_val):
    # Convert all object columns to string
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    # One-hot encode all categorical features
    combined = pd.concat([X_train, X_val], axis=0)
    combined_encoded = pd.get_dummies(combined)

    # Split back into train and val sets
    X_train_encoded = combined_encoded.iloc[:len(X_train), :].copy()
    X_val_encoded = combined_encoded.iloc[len(X_train):, :].copy()

    return X_train_encoded, X_val_encoded


def load_data(train_path, val_path, label_col="Price_euros"):
    try:
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
    except Exception as e:
        print(f"Error reading CSV files:\n{e}")
        raise

    # Optional: Strip spaces and unify column names
    train.rename(columns=lambda x: x.strip(), inplace=True)
    val.rename(columns=lambda x: x.strip(), inplace=True)

    # Log-transform target
    train[label_col] = np.log1p(train[label_col])
    val[label_col] = np.log1p(val[label_col])

    X_train = train.drop(columns=[label_col])
    y_train = train[label_col]
    X_val = val.drop(columns=[label_col])
    y_val = val[label_col]

    return X_train, y_train, X_val, y_val


def scale_numeric(X_train, X_val):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return X_train, X_val


# ========================
# Plotting Feature Importance
# ========================

def plot_feature_importance(model, feature_names, model_type, top_n=15):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        top_features = np.array(feature_names)[sorted_idx][:top_n]
        top_importance = importance[sorted_idx][:top_n]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_importance)), top_importance, align='center')
        plt.yticks(range(len(top_importance)), top_features)
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Features - {model_type}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{model_type}_top{top_n}_feature_importance.png")
        plt.close()
        print(f"Saved plot: {model_type}_top{top_n}_feature_importance.png")

        feature_importance = pd.Series(model.feature_importances_, index=feature_names)
        print(f"\nTop {top_n} Features for {model_type}:\n")
        print(feature_importance.sort_values(ascending=False).head(top_n))
    else:
        print(f"Feature importance not available for {model_type}")


# ========================
# Training + Evaluation
# ========================

def train_and_evaluate(X_train, y_train, X_val, y_val, model_type="linear", do_grid_search=True):
    if model_type == "linear":
        model = LinearRegression()

    elif model_type == "random_forest":
        model = RandomForestRegressor(random_state=42)
        if do_grid_search:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            }
            model = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42)
        if do_grid_search:
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            }
            model = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

    elif model_type == "xgboost":
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        if do_grid_search:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0]
            }
            model = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    final_model = model.best_estimator_ if isinstance(model, GridSearchCV) else model

    y_pred = final_model.predict(X_val)
    y_pred = np.expm1(y_pred)
    y_val_orig = np.expm1(y_val)

    mse = mean_squared_error(y_val_orig, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_orig, y_pred)
    r2 = r2_score(y_val_orig, y_pred)

    print(f"\nModel: {model_type}")
    if isinstance(model, GridSearchCV):
        print(f"Best Params: {model.best_params_}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    plot_feature_importance(final_model, X_train.columns, model_type)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val_orig, y_pred, alpha=0.5)
    plt.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--')
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title(f"{model_type} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_type}_actual_vs_predicted.png")
    plt.close()
    print(f"Saved plot: {model_type}_actual_vs_predicted.png")

    return final_model


# ========================
# Main Driver
# ========================

def main():
    X_train, y_train, X_val, y_val = load_data("data/laptop-data-train.csv", "data/laptop-data-val.csv")

    # Sanitize column names
    X_train.columns = X_train.columns.str.replace(r"[<>[\]]", "_", regex=True)
    X_val.columns = X_val.columns.str.replace(r"[<>[\]]", "_", regex=True)

    for model_type in ["linear", "random_forest", "gradient_boosting"]:
        print(f"\n=== {model_type.upper()} ===")

        # Use one-hot encoding for xgboost only
        if model_type == "xgboost":
            X_train_encoded, X_val_encoded = preprocess_for_xgboost(X_train.copy(), X_val.copy())
            X_train_scaled, X_val_scaled = scale_numeric(X_train_encoded, X_val_encoded)
        else:
            X_train_scaled, X_val_scaled = scale_numeric(X_train.copy(), X_val.copy())

        model = train_and_evaluate(X_train_scaled, y_train, X_val_scaled, y_val, model_type)
        joblib.dump(model, f"laptop_price_model_{model_type}.joblib")
        print(f"Saved {model_type} model\n")



if __name__ == "__main__":
    main()
