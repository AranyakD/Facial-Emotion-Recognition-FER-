from tensorflow.keras.applications import efficientnet
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import matplotlib.pyplot as plt
import os

train_dir = "dataset_Train"
test_dir = "dataset_Test"
img_height = 224
img_width = 224
batch_size = 32

# loading pretrained EfficientNetB0
base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# freezing model
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

# loading train dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    shuffle=True
)

# loading test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb"
)

""" # verifying class labels
class_names = train_dataset.class_names
print("Emotion Classes:", class_names) """

""" for images, labels in train_dataset.take(1):
    print("min pixel val:", tf.reduce_min(images))
    print("max pixel val:", tf.reduce_max(images)) """

# data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

# preprocessing
preprocess_input = efficientnet.preprocess_input

train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y))

# caching
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# performance opt
autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=autotune)
test_dataset = test_dataset.prefetch(buffer_size=autotune)

# Defining model
model = tf.keras.models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.BatchNormalization(),

    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(7, activation='softmax')
])

model.summary()

# compling the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# fixing overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# training the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    callbacks=[early_stop]
)

# checking for improvements
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()
