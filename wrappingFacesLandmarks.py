import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import glob
from scipy.spatial import Delaunay
import pyaudio
import wave
import threading
from moviepy import VideoFileClip, AudioFileClip

# ============================================================
#                    CONFIGURATION
# ============================================================
FRAME_W, FRAME_H = 640, 480
TARGET_FOLDER = "Targets"
WINDOW_NAME = "Real-time Morph"

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
#            LOAD LANDMARKS FROM JSON
# ============================================================
def load_landmarks_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    pts = np.array(data["points"], dtype=np.int32)
    w = int(data["width"])
    h = int(data["height"])
    return pts, w, h


# ============================================================
#          MAKE CIRCULAR ICON FOR FILTER BAR
# ============================================================
def make_circle_icon(img, size=70):
    # Convert to BGR if it has alpha channel
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    icon = cv2.resize(img, (size, size))
    mask = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(mask, (size//2, size//2), size//2, (1,1,1), -1)
    return (icon * mask).astype(np.uint8)


# ============================================================
#     WARP ONE TRIANGLE (target → source triangle)
# ============================================================
def warp_triangle(img_src, img_dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    
    # Safety checks: skip if rectangles are invalid or out of bounds
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return
    if r1[0] < 0 or r1[1] < 0 or r1[0]+r1[2] > img_src.shape[1] or r1[1]+r1[3] > img_src.shape[0]:
        return
    if r2[0] < 0 or r2[1] < 0 or r2[0]+r2[2] > img_dst.shape[1] or r2[1]+r2[3] > img_dst.shape[0]:
        return

    t1_rect = []
    t2_rect = []

    for i in range(3):
        t1_rect.append((t_src[i][0] - r1[0], t_src[i][1] - r1[1]))
        t2_rect.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))

    src_crop = img_src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    
    # Additional safety check for cropped region
    if src_crop.size == 0:
        return

    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv2.warpAffine(src_crop, M, (r2[2], r2[3]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1,1,1), 16, 0)

    dst_area = img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]].astype(np.float32)
    
    # Final safety check: ensure shapes match before blending
    if dst_area.shape != warped.shape or dst_area.shape != mask.shape:
        return
    
    dst_area = dst_area * (1 - mask) + warped.astype(np.float32) * mask
    img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = np.clip(dst_area, 0, 255).astype(np.uint8)


# ============================================================
#         LOAD ALL TARGET IMAGES + JSON INTO LIST
# ============================================================
target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.png")))
targets = []

for f in target_files:
    name = os.path.splitext(os.path.basename(f))[0]
    json_path = os.path.join(TARGET_FOLDER, name + ".json")

    if not os.path.exists(json_path):
        print(f"WARNING: JSON for {name} not found!")
        continue

    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
     
    # Convert RGBA to BGR if needed
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    pts, w, h = load_landmarks_json(json_path)
    img = cv2.resize(img, (w, h))

    icon = make_circle_icon(img)

    targets.append({
        "name": name,
        "img": img,
        "pts": pts,
        "icon": icon
    })

if len(targets) == 0:
    raise Exception("No targets found in folder.")

print("Loaded targets:", [t["name"] for t in targets])

active_target_index = 0
active_target = targets[active_target_index]

# Build Delaunay triangulation ONCE using first target
tri = Delaunay(active_target["pts"])
triangles = tri.simplices


# ============================================================
#                 OPEN WEBCAM + MESH
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=False,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


# ============================================================
#                  MOUSE CLICK HANDLER
# ============================================================
def mouse_event(event, x, y, flags, param):
    global active_target_index, active_target

    if event == cv2.EVENT_LBUTTONDOWN:
        bar_y = FRAME_H - 90
        if y >= bar_y:
            icon_w = 80
            for i in range(len(targets)):
                x0 = 20 + i * icon_w
                x1 = x0 + 70
                if x0 <= x <= x1:
                    active_target_index = i
                    active_target = targets[i]
                    print("Switched to:", active_target["name"])
                    return

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_event)

cv2.createTrackbar("alpha", WINDOW_NAME, 50, 100, lambda x: None)

# Video recording variables
is_recording = False
video_writer = None
output_filename = ""

# Audio recording variables
audio_frames = []
audio_thread = None
audio_stream = None
p_audio = pyaudio.PyAudio()

# Audio recording function (runs in separate thread)
def record_audio():
    global audio_frames, audio_stream
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    audio_stream = p_audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)
    
    audio_frames = []
    while is_recording:
        try:
            data = audio_stream.read(CHUNK, exception_on_overflow=False)
            audio_frames.append(data)
        except:
            break


# ============================================================
#                       MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    display = frame.copy()

    if results.multi_face_landmarks:
        # Process each detected face
        for face_lms in results.multi_face_landmarks:

            src_pts = []
            for lm in face_lms.landmark:
                src_pts.append([int(lm.x * FRAME_W), int(lm.y * FRAME_H)])
            src_pts = np.array(src_pts, dtype=np.int32)

            target_img = active_target["img"]
            target_pts = active_target["pts"]

            warped_target = np.zeros_like(frame)

            for tri_idx in triangles:
                t_tgt = target_pts[tri_idx]
                t_src = src_pts[tri_idx]
                warp_triangle(target_img, warped_target, t_tgt, t_src)

            # Create face mask to only blend face area
            face_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            hull = cv2.convexHull(src_pts)
            cv2.fillConvexPoly(face_mask, hull, 255)
            
            # Create teeth/mouth interior mask - always exclude to show real teeth
            # Inner mouth indices for teeth area
            teeth_indices = [
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,  # Inner mouth contour
                95, 88, 178, 87, 14, 317, 402, 318, 324, 308  # Additional inner area
            ]
            
            # Eye indices for both eyes
            left_eye_indices = [
                # Left eye contour
                33, 7, 163, 144, 145, 153, 154, 155, 133,
                173, 157, 158, 159, 160, 161, 246
            ]
            
            right_eye_indices = [
                # Right eye contour
                362, 382, 381, 380, 374, 373, 390, 249,
                263, 466, 388, 387, 386, 385, 384, 398
            ]
            
            if len(src_pts) > max(teeth_indices):
                teeth_pts = src_pts[teeth_indices]
                teeth_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
                cv2.fillConvexPoly(teeth_mask, cv2.convexHull(teeth_pts), 255)
                
                # Expand teeth mask to ensure full teeth coverage
                kernel = np.ones((3,3), np.uint8)
                teeth_mask = cv2.dilate(teeth_mask, kernel, iterations=1)
                
                # Remove teeth area from face mask (so real teeth always show)
                face_mask = cv2.subtract(face_mask, teeth_mask)
            
            # Exclude eyes area to show real eyes
            if len(src_pts) > max(left_eye_indices + right_eye_indices):
                # Left eye mask
                left_eye_pts = src_pts[left_eye_indices]
                left_eye_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
                cv2.fillConvexPoly(left_eye_mask, cv2.convexHull(left_eye_pts), 255)
                
                # Right eye mask
                right_eye_pts = src_pts[right_eye_indices]
                right_eye_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
                cv2.fillConvexPoly(right_eye_mask, cv2.convexHull(right_eye_pts), 255)
                
                # Combine eye masks
                eyes_mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
                
                # Expand eyes mask to ensure full coverage
                kernel = np.ones((5,5), np.uint8)
                eyes_mask = cv2.dilate(eyes_mask, kernel, iterations=2)
                
                # Remove eyes area from face mask (so real eyes always show)
                face_mask = cv2.subtract(face_mask, eyes_mask)
            
            # Blur mask edges for smooth blending
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 11)
            face_mask_3ch = cv2.merge([face_mask, face_mask, face_mask]) / 255.0
            
            # Alpha blend
            alpha = cv2.getTrackbarPos("alpha", WINDOW_NAME) / 100
            face_blend = cv2.addWeighted(display, 1 - alpha, warped_target, alpha, 0)
            
            # Combine: blended face in mask area, original frame (including mouth) elsewhere
            display = (face_blend * face_mask_3ch + display * (1 - face_mask_3ch)).astype(np.uint8)
            
            # Force original eyes and mouth to show (no blending whatsoever)
            if len(src_pts) > max(left_eye_indices + right_eye_indices):
                # Create sharp (non-blurred) masks for eyes and mouth
                final_exclusion_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
                
                # Add eyes
                cv2.fillConvexPoly(final_exclusion_mask, cv2.convexHull(left_eye_pts), 255)
                cv2.fillConvexPoly(final_exclusion_mask, cv2.convexHull(right_eye_pts), 255)
                
                # Add mouth if available
                if len(src_pts) > max(teeth_indices):
                    cv2.fillConvexPoly(final_exclusion_mask, cv2.convexHull(teeth_pts), 255)
                
                # Expand to ensure full coverage
                kernel = np.ones((5,5), np.uint8)
                final_exclusion_mask = cv2.dilate(final_exclusion_mask, kernel, iterations=2)
                
                # Convert to 3 channels
                final_exclusion_mask_3ch = cv2.merge([final_exclusion_mask, final_exclusion_mask, final_exclusion_mask]) / 255.0
                
                # Replace with original frame in these areas (100% original, 0% morphed)
                display = (frame * final_exclusion_mask_3ch + display * (1 - final_exclusion_mask_3ch)).astype(np.uint8)

    # Write frame BEFORE adding UI elements (so they don't appear in the video)
    if is_recording and video_writer is not None:
        video_writer.write(display)
    
    # ------------ DRAW ICON BAR (after recording frame) ------------
    icon_y = FRAME_H - 90
    x0 = 20
    for i, t in enumerate(targets):
        center_x = x0 + 35  # Center of 70px icon
        center_y = icon_y + 35
        
        # Draw selection circle
        if i == active_target_index:
            cv2.circle(display, (center_x, center_y), 38, (0, 255, 255), 3)
        
        # Create circular mask for icon
        icon_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cv2.circle(icon_mask, (center_x, center_y), 35, 255, -1)
        
        # Apply icon with circular mask
        icon_region = display[icon_y:icon_y+70, x0:x0+70]
        icon_mask_region = icon_mask[icon_y:icon_y+70, x0:x0+70]
        icon_mask_3ch = cv2.merge([icon_mask_region, icon_mask_region, icon_mask_region]) / 255.0
        
        icon_region[:] = (t["icon"] * icon_mask_3ch + icon_region * (1 - icon_mask_3ch)).astype(np.uint8)
        
        x0 += 80
    
    # Add recording indicator AFTER writing frame (only shows in window)
    if is_recording:
        cv2.circle(display, (FRAME_W - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(display, "REC", (FRAME_W - 70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break
    elif key == ord('r'):  # R to toggle recording
        if not is_recording:
            # Start recording
            output_filename = "temp_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_filename, fourcc, 20.0, (FRAME_W, FRAME_H))
            is_recording = True
            
            # Start audio recording in separate thread
            audio_thread = threading.Thread(target=record_audio)
            audio_thread.start()
            
            print(f"🔴 Recording started (with audio)")
        else:
            # Stop recording
            is_recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            
            # Wait for audio thread to finish
            if audio_thread is not None:
                audio_thread.join()
            
            # Save audio to WAV file
            if audio_stream is not None:
                audio_stream.stop_stream()
                audio_stream.close()
            
            print("⏹️  Recording stopped. Processing...")
            
            # Create Recordings folder if it doesn't exist
            os.makedirs("Recordings", exist_ok=True)
            
            if len(audio_frames) > 0:
                # Save audio temporarily
                audio_filename = "temp_audio.wav"
                print(f"Saving audio to {audio_filename}...")
                wf = wave.open(audio_filename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p_audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(audio_frames))
                wf.close()
                print(f"Audio saved. Size: {os.path.getsize(audio_filename)} bytes")
                
                # Merge audio and video into single file
                print("Merging audio and video...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                final_output = os.path.join("Recordings", f"morph_{timestamp}.mp4")
                
                try:
                    video_clip = VideoFileClip(output_filename)
                    print(f"Video loaded: {video_clip.duration}s, {video_clip.fps} fps")
                    
                    audio_clip = AudioFileClip(audio_filename)
                    print(f"Audio loaded: {audio_clip.duration}s")
                    
                    final_clip = video_clip.with_audio(audio_clip)
                    print(f"Writing merged video to {final_output}...")
                    
                    final_clip.write_videofile(final_output, codec='libx264', audio_codec='aac', logger=None)
                    
                    # Clean up clips
                    final_clip.close()
                    video_clip.close()
                    audio_clip.close()
                    
                    # Delete temporary files
                    print("Cleaning up temporary files...")
                    os.remove(output_filename)
                    os.remove(audio_filename)
                    
                    print(f"✅ Video with audio saved: {final_output}")
                except Exception as e:
                    import traceback
                    print(f"⚠️  Audio merge failed!")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    
                    # Move video-only file to Recordings folder
                    final_output = os.path.join("Recordings", f"morph_{timestamp}_no_audio.mp4")
                    if os.path.exists(output_filename):
                        try:
                            os.rename(output_filename, final_output)
                            print(f"✅ Video saved (no audio): {final_output}")
                        except PermissionError:
                            print(f"⚠️  Could not move file, it may still be in use")
                            print(f"Video is at: {output_filename}")
            else:
                # No audio, just move video to Recordings folder
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                final_output = os.path.join("Recordings", f"morph_{timestamp}_no_audio.mp4")
                os.rename(output_filename, final_output)
                print(f"✅ Video saved (no audio captured): {final_output}")

# Clean up
if is_recording and video_writer is not None:
    video_writer.release()
    print(f"⏹️  Recording stopped: {output_filename}")

p_audio.terminate()
cap.release()
cv2.destroyAllWindows()
print("Program closed.")
