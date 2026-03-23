import cv2
import numpy as np
from tensorflow.keras.models import load_model


# loading trained model
model = load_model("models/final_FER_model.keras")

# emotion labels
emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# loading Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascasde_frontalface_default.xml'
)

# starting webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # resize to match training
        face = cv2.resize(face, (192, 192))

        face = np.array(face, dtype='float32')
        face = np.expand_dims(face, axis=0)

        # preprocessing
        face = tf.keras.applications.efficientnet.preprocess_input(face)

        # prediction
        preds = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]

        # draw
        cv2, rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('FER - Emotion Detection', frame)

    if cv2.waitkey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
