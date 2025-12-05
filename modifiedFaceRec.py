import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Minimal initialization
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,  # Set to False if having issues
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Face Landmark Detection Started")
print("Press ESC to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = face_mesh.process(rgb_frame)
    
    # Draw landmarks as simple points
    if results.multi_face_landmarks:
        h, w = frame.shape[:2]
        
        for face_landmarks in results.multi_face_landmarks:
            # Draw all 468 landmarks
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Draw different colors for different features
                if idx < 468:  # All face points
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Highlight key points (optional)
                # Nose tip: 1, Left eye: 133, Right eye: 362, Mouth: 13, 14
                if idx in [1, 133, 362, 13, 14]:
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        # Display landmark count
        cv2.putText(frame, f"Landmarks: {len(face_landmarks.landmark)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Face Landmarks', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("Closed successfully")