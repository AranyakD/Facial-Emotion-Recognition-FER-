# Facial Emotion Recognition (FER)

## Overview

This project implements a real-time Facial Emotion Recognition system using a deep learning model based on EfficientNet. The application captures live video from a webcam, detects faces, and predicts human emotions with confidence scores.
The model used primarily is a transfer learning model (EfficientNetB0) but before that I did develop a baseline CNN model from scratch which had around 50% accuracy. 

The system also displays the probability distribution across all emotion classes, providing a transparent and interpretable output.

---

## Features

* Real-time face detection using OpenCV
* Deep learning model based on EfficientNet (Transfer Learning)
* Baseline CNN file is also included in the repo.
* Emotion classification into 7 categories:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Sad
  * Surprise
  * Neutral

* Confidence score display
* Emotion probability distribution (all classes)
* Prediction smoothing to reduce flickering
* Low-confidence detection ("Uncertain" state)

---

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

---

## Project Structure
(main structure used for running the model, I have uploaded the other contributing files regardless that arent used in the primary model)

```
Facial-Emotion-Recognition/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── FER_FINAL.keras
├── assets/
│   └── demo.png
```

---

## Installation

1. Clone the repository:

```
git clone https://github.com/AranyakD/Facial-Emotion-Recognition-FER-.git
cd Facial-Emotion-Recognition
```

2. Create a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

Run the following command:

```
python app.py
```

Press `ESC` to exit the application.

---

## Model Details

* Architecture: EfficientNetB0 (Transfer Learning)
* Input size: 192x192 RGB images
* Output: 7 emotion classes (Softmax)
* Preprocessing: `tf.keras.applications.efficientnet.preprocess_input`

---

![Demo](assets/demo.png)

---

## Future Improvements

* Improve model accuracy with better datasets
* Add support for multiple face tracking
* Deploy as a web application (Flask/Streamlit)
* Optimize inference speed for edge devices

---

## License

This project is for academic and educational purposes.
