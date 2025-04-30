import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Embedding, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def clean_ram_and_weight(df):
    df = df.copy()

    if 'Ram' in df.columns:
        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).replace('nan', np.nan)
        df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce').fillna(0).astype(int)

    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].astype(str).str.replace('kg', '', regex=False).replace('nan', np.nan)
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(0.0).astype(float)

    return df

def load_and_preprocess(train_path, val_path, label_col="Price_euros"):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    train = clean_ram_and_weight(train)
    val = clean_ram_and_weight(val)

    # Select features
    categorical_features = ['Company', 'Product', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
    numeric_features = ['Inches', 'Ram', 'Weight']

    # Convert categorical features to categories
    for col in categorical_features:
        all_vals = pd.concat([train[col], val[col]])
        categories = all_vals.astype('category').cat.categories
        train[col] = pd.Categorical(train[col], categories=categories).codes
        val[col] = pd.Categorical(val[col], categories=categories).codes

    X_train = train[categorical_features + numeric_features]
    X_val = val[categorical_features + numeric_features]
    y_train = np.log1p(train[label_col])
    y_val = np.log1p(val[label_col])

    return X_train, y_train, X_val, y_val, categorical_features, numeric_features


def build_model(categorical_features, numeric_features, X_train):
    inputs = []
    encoded_features = []

    for cat_col in categorical_features:
        input_cat = Input(shape=(1,), name=cat_col)
        vocab_size = X_train[cat_col].nunique() + 1
        embed_dim = min(50, (vocab_size // 2) + 1)
        embed = Embedding(input_dim=vocab_size, output_dim=embed_dim)(input_cat)
        flat = Flatten()(embed)
        inputs.append(input_cat)
        encoded_features.append(flat)

    input_numeric = Input(shape=(len(numeric_features),), name="numeric")
    norm = BatchNormalization()(input_numeric)
    inputs.append(input_numeric)
    encoded_features.append(norm)

    x = Concatenate()(encoded_features)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    X_train, y_train, X_val, y_val, cat_feats, num_feats = load_and_preprocess(
        "data/laptop-data-train.csv", "data/laptop-data-val.csv"
    )

    model = build_model(cat_feats, num_feats, X_train)

    train_input = {col: X_train[col].values for col in cat_feats}
    train_input["numeric"] = X_train[num_feats].values
    val_input = {col: X_val[col].values for col in cat_feats}
    val_input["numeric"] = X_val[num_feats].values

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_input, y_train,
        validation_data=(val_input, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_log = model.predict(val_input).flatten()
    y_pred = np.expm1(y_pred_log)
    y_val_orig = np.expm1(y_val)

    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
    mae = mean_absolute_error(y_val_orig, y_pred)
    r2 = r2_score(y_val_orig, y_pred)

    print(f"\nDeeper Neural Network Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val_orig, y_pred, alpha=0.5)
    plt.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--')
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title("Deep NN - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("deep_nn_actual_vs_predicted.png")
    plt.close()
    print("Saved plot: deep_nn_actual_vs_predicted.png")

    model.save("deep_nn_model.keras")
    print("Model and preprocessor saved!")


if __name__ == "__main__":
    main()
