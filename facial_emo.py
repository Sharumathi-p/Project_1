import cv2
from deepface import DeepFace
import time

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

# Define emotion labels including panic
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'panic']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Debug: Print the number of faces detected
    print(f"Number of faces detected: {len(faces)}")

    # Process each face
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        try:
            # Analyze the emotion in the detected face
            result = DeepFace.analyze(img_path=roi_color, actions=['emotion'], enforce_detection=False)
            emotion = result['dominant_emotion']

            # Map 'fear' to 'panic' for better understanding
            if emotion == 'fear':
                emotion = 'panic'

        except Exception as e:
            emotion = "N/A"
            # Debug: Print the exception message
            print(f"Exception occurred: {e}")

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Calculate and overlay FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the frame with overlays
    cv2.imshow("Real-time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
