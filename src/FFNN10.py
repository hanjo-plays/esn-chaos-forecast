from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent

data_file = PROJECT_ROOT / "data" / "lorenz_windows_w10.csv"

# Read the CSV
df = pd.read_csv(data_file)
print(df.head(5))


WINDOW = 10
feature_col = [f"{axis}_t{i}" for i in range(WINDOW) for axis in ("x", "y", "z")]

target_col = [c for c in df.columns if c.endswith("_next")]

X = df[feature_col].values
y = df[target_col].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

batch_size = 32


def make_dataset(X_arr, y_arr, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X_arr))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


ds_train = make_dataset(X_train, y_train)
ds_val = make_dataset(X_val, y_val, shuffle=False)


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


WINDOW = 10
input_dim = WINDOW * 3

# Build the Sequential model
model_seq = build_ffnn_full_seq(input_dim)

# Adapt the normalization layer (must do *after* model is built)
norm_layer = model_seq.get_layer("norm")
norm_layer.adapt(ds_train.map(lambda X, y: X))

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
        "ffnn_full_seq_w10_best.h5", save_best_only=True, monitor="val_loss"
    ),
]

# Train
history_seq = model_seq.fit(
    ds_train, validation_data=ds_val, epochs=200, callbacks=callbacks
)

# Evaluate
val_loss, val_mae = model_seq.evaluate(ds_val)
print(f"Seq model — loss: {val_loss:.4f}, mae: {val_mae:.4f}")

# 1) Gather y_true and y_pred from ds_val
y_true = np.vstack([y for x, y in ds_val])
y_pred = model_seq.predict(ds_val)

# 2) RMSE
rmse = root_mean_squared_error(y_true, y_pred)
print(f"Seq model — RMSE: {rmse:.4f}")

# 3) R² score
r2 = r2_score(y_true, y_pred)
print(f"Seq model — R²: {r2:.4f}")
