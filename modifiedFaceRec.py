import cv2 
import mediapipe as mp 
import json
import time
import os

# Create TargetLandmarks folder if it doesn't exist
os.makedirs("TargetLandmarks", exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh 

face_mesh = mp_face_mesh.FaceMesh( 
    max_num_faces=5,  # Detect up to 5 faces
    refine_landmarks=False,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5 
) 

cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

print("Face Landmark Detection Started") 
print("Press ESC to quit")
print("Press 's' to save SOURCE landmarks (your current face)")
print("Press 't' to save TARGET landmarks (real-time target)")

while cap.isOpened(): 
    ret, frame = cap.read() 
    if not ret: 
        break 
     
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = face_mesh.process(rgb_frame) 
     
    if results.multi_face_landmarks: 
        h, w = frame.shape[:2]
        
        # Draw landmarks for ALL detected faces
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark: 
                x = int(lm.x * w) 
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        face_landmarks = results.multi_face_landmarks[0]  # Save first face for S/T keys
        cv2.putText(frame, f"Faces detected: {len(results.multi_face_landmarks)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    else:
        face_landmarks = None
        cv2.putText(frame, "No face detected", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow('Face Landmarks', frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC to quit
    if key == 27:
        break

    # ---- SAVE SOURCE WITH 'S' ----
    if key == ord('s'):
        if face_landmarks:
            pts = []
            for lm in face_landmarks.landmark:
                pts.append([int(lm.x * w), int(lm.y * h)])

            data = {
                "type": "source",
                "timestamp": time.time(),
                "width": w,
                "height": h,
                "points": pts
            }

            with open("source_landmarks.json", "w") as f:
                json.dump(data, f, indent=2)

            print("✅ SOURCE saved → source_landmarks.json")
        else:
            print("⚠ No face detected to save as SOURCE.")

    # ---- SAVE TARGET WITH 'T' ----
    if key == ord('t'):
        if face_landmarks:
            pts = []
            for lm in face_landmarks.landmark:
                pts.append([int(lm.x * w), int(lm.y * h)])

            data = {
                "type": "target",
                "timestamp": time.time(),
                "width": w,
                "height": h,
                "points": pts
            }

            output_path = os.path.join("TargetLandmarks", "target_landmarks.json")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"🎯 TARGET saved → {output_path}")
        else:
            print("⚠ No face detected to save as TARGET.")

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
