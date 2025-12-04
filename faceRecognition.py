import cv2
import numpy as np

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load facial landmark detector
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Detect landmarks
    if len(faces) > 0:
        ok, landmarks = facemark.fit(gray, faces)
        
        if ok:
            # Draw facial landmarks as points
            for landmark in landmarks:
                for point in landmark[0]:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Show the result
    cv2.imshow("Face Landmarks", frame)
    
    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()