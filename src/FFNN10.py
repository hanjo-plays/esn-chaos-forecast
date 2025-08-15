from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.keras  # works with tf.keras


PROJECT_ROOT = Path(__file__).parent.parent

data_file = PROJECT_ROOT / "data" / "lorenz_windows_w10.csv"

# store runs inside your repo under mlruns/
mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
mlflow.set_experiment("Lorenz_FFNN")
mlflow.keras.autolog(log_models=False)

# Read the CSV
df = pd.read_csv(data_file)
print(df.head(5))


WINDOW = 10
feature_col = [f"{axis}_t{i}" for i in range(WINDOW) for axis in ("x", "y", "z")]

target_col = ["x_next", "y_next", "z_next"]

X = df[feature_col].values
y = df[target_col].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

batch_size = 32

def build_ffnn_full_seq(input_dim: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="history_input"),
            tf.keras.layers.Normalization(name="norm"),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer="l2"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer="l2"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, name="predictions"),
        ],
        name="ffnn_full_seq_w10",
    )

input_dim = WINDOW * 3

# Build the Sequential model
model_seq = build_ffnn_full_seq(input_dim)

# Adapt the normalization layer (must do *after* model is built)
norm_layer = model_seq.get_layer("norm")
norm_layer.adapt(X_train.astype("float32"))

# Compile
model_seq.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])

# Callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir="logs/ffnn_w10",
        histogram_freq=1,  # record weight & activation histograms every epoch
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "src/ffnn_full_seq_w10_best.h5", save_best_only=True, monitor="val_loss"
    ),
]

# Train
history_seq = model_seq.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=200,
    callbacks=callbacks,
    shuffle=True
)

# Evaluate
val_loss, val_mae = model_seq.evaluate(X_val, y_val, batch_size=batch_size)
print(f"Seq model — loss: {val_loss:.4f}, mae: {val_mae:.4f}")

# Predictions
y_true = y_val
y_pred = model_seq.predict(X_val, batch_size=batch_size)

# RMSE
rmse = root_mean_squared_error(y_true, y_pred)
print(f"Seq model — RMSE: {rmse:.4f}")

# R² score
r2 = r2_score(y_true, y_pred)
print(f"Seq model — R²: {r2:.4f}")
