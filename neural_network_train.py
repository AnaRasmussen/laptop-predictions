import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.callbacks import EarlyStopping


def load_and_preprocess(train_path, val_path, label_col="Price_euros"):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    y_train = train[label_col]
    y_val = val[label_col]
    X_train = train.drop(columns=[label_col])
    X_val = val.drop(columns=[label_col])

    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocess", preprocessor)
    ])

    X_train_processed = pipeline.fit_transform(X_train)
    X_val_processed = pipeline.transform(X_val)

    return X_train_processed, np.log1p(y_train), X_val_processed, np.log1p(y_val), pipeline


def build_model(input_shape):
    tf.random.set_seed(42)
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_shape,)))

    # Layer 1
    model.add(keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    # Layer 2
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # Layer 3
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # Output layer
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

    # model = Sequential([
    #     Input(shape=(input_shape,)),

    #     Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    #     BatchNormalization(),
    #     Dropout(0.2),

    #     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    #     BatchNormalization(),
    #     Dropout(0.2),

    #     Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    #     BatchNormalization(),
    #     Dropout(0.1),

    #     Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    #     BatchNormalization(),

    #     Dense(1)
    # ])

    # model.compile(
    #     optimizer=Adam(learning_rate=0.001),
    #     loss='mse',
    #     metrics=['mae']
    # )
    # return model


def plot_loss(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Neural Network Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("nn_loss_curve.png")
    plt.close()
    print("Saved plot: nn_loss_curve.png")


def plot_predictions(y_true, y_pred):
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title("NN - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("nn_actual_vs_predicted.png")
    plt.close()
    print("Saved plot: nn_actual_vs_predicted.png")


def main():
    X_train, y_train_log, X_val, y_val_log, pipeline = load_and_preprocess(
        "data/laptop-data-train.csv", "data/laptop-data-val.csv"
    )

    model = build_model(X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train_log,
        validation_data=(X_val, y_val_log),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    plot_loss(history)

    y_pred_log = model.predict(X_val).flatten()
    y_pred = np.expm1(y_pred_log)
    y_val = np.expm1(y_val_log)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\nNeural Network Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    plot_predictions(y_val, y_pred)
    model.save("laptop_price_nn_model.keras")
    joblib.dump(pipeline, "nn_preprocessor.joblib")
    print("Saved model: laptop_price_nn_model.keras")
    print("Saved preprocessor: nn_preprocessor.joblib")

if __name__ == "__main__":
    main()