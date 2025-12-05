import cv2
import numpy as np

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load facial landmark detector
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

# Open camera
cap = cv2.VideoCapture(0)

# Reduce resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Try to disable auto-exposure and adjust brightness
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto-exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Reduce exposure for bright light
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Adjust brightness

# Frame counter for face detection optimization
frame_count = 0
faces = []
previous_landmarks = None  # Store previous landmarks for smoothing

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast in varying light
    gray = cv2.equalizeHist(gray)
    
    # Only detect faces every 3 frames to reduce lag
    if frame_count % 3 == 0:
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # More sensitive detection
            minNeighbors=3,    # Less strict (was 5)
            minSize=(100, 100) # Minimum face size
        )
    
    frame_count += 1
    
    # Detect landmarks
    if len(faces) > 0:
        ok, landmarks = facemark.fit(gray, faces)
        
        if ok:
            current_landmarks = landmarks[0][0]
            
            # Smooth landmarks by averaging with previous frame
            if previous_landmarks is not None:
                current_landmarks = 0.7 * current_landmarks + 0.3 * previous_landmarks
            
            previous_landmarks = current_landmarks.copy()
            
            # Draw facial landmarks as points
            for point in current_landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    else:
        previous_landmarks = None  # Reset if no face detected
    
    # Show the result
    cv2.imshow("Face Landmarks", frame)
    
    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()