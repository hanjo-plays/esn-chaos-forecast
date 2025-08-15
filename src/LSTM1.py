from pathlib import Path

import mlflow
import mlflow.keras  # works with tf.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent

data_file = PROJECT_ROOT / "data" / "lorenz_windows_w10.csv"

# store runs inside your repo under mlruns/
mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
mlflow.set_experiment("Lorenz_LSTM")
mlflow.keras.autolog(log_models=False)

# Read the CSV
df = pd.read_csv(data_file)
print(df.head(5))


feature_col = ["x_t0", "y_t0", "z_t0"]
target_col = ["x_next", "y_next", "z_next"]

X = df[feature_col].values
y = df[feature_col].values

X = X.reshape(-1, 1, 3)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
)

batch_size = 32


def build_lstm_w1_seq() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1, 3), name="last_state"),  # (time=1, feat=3)
            tf.keras.layers.Normalization(name="norm"),
            tf.keras.layers.LSTM(64, name="lstm"),  # -> (batch, 64)
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, name="next_state"),
        ],
        name="lstm_w1_seq",
    )


# Build the Sequential model
model = build_lstm_w1_seq()

# IMPORTANT: adapt the Normalization layer on *training inputs*
norm = model.get_layer("norm")
norm.adapt(X_train)  # shape (N, 1, 3) — matches model's expected input rank

# compile
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

# callbacks (optional but good practice)
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir="logs/lstm_w1",
        histogram_freq=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "src/lstm_w1_best.h5", save_best_only=True, monitor="val_loss"
    ),
]

# train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=batch_size,
    callbacks=callbacks,
    shuffle=True,
)

# evaluate + extra metrics
val_mse, val_mae = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)

# Predictions
y_true = y_val
y_pred = model.predict(X_val, batch_size=batch_size, verbose=0)

# Print results
print(f"Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}")

# RMSE
rmse = root_mean_squared_error(y_true, y_pred)
print(f"Seq model — RMSE: {rmse:.4f}")

# R² score
r2 = r2_score(y_true, y_pred)
print(f"Seq model — R²: {r2:.4f}")
