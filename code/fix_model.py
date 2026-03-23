import tensorflow as tf
import os

# path handling
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "final_FER_model.keras")

print("Looking for model at:", model_path)
print("Exists?", os.path.exists(model_path))

print("Loading broken model")

model = tf.keras.models.load_model(
    model_path,
    compile=False,
    safe_mode=False
)

print("Rebuilding model")

inputs = tf.keras.Input(shape=(192, 192, 3))
outputs = model(inputs)
fixed_model = tf.keras.Model(inputs, outputs)

print("Saving fixed model")

fixed_path = os.path.join(BASE_DIR, "models", "fixed_model.keras")
fixed_model.save(fixed_path)

print("Done. Model repaired successfully.")
