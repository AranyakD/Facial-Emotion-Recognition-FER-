import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os


# loading model
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "FER_FINAL.keras")

print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully")

# labels
emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# webcam
cap = cv2.VideoCapture(0)

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # resize to match training
        face = cv2.resize(face, (192, 192))

        face = np.array(face, dtype='float32')
        face = np.expand_dims(face, axis=0)

        # EfficientNet preprocessing
        face = tf.keras.applications.efficientnet.preprocess_input(face)

        # prediction
        preds = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]

        # draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('FER - Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
