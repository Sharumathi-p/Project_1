import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained emotion detection model
model = load_model("emotion_model.h5")

# Define emotion labels (ensure the order matches your training labels)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)
prev_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Calculate and overlay FPS
    curr_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        try:
            # Preprocess face ROI to match training: grayscale, resize, normalize
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized.astype("float") / 255.0
            roi_expanded = np.expand_dims(np.expand_dims(roi_normalized, -1), 0)  # Shape: (1, 48, 48, 1)
            
            # Predict emotion
            prediction = model.predict(roi_expanded)
            emotion = emotion_labels[np.argmax(prediction)]
        except Exception as e:
            emotion = "N/A"
            print(f"Exception occurred: {e}")
        
        # Draw a rectangle around the face and display the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    cv2.imshow("Live Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
