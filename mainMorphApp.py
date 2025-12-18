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
import ctypes


# ============================================================
#                    CONFIGURATION
# ============================================================
TARGET_FOLDER = "Targets"
WINDOW_NAME = "Real-time Morph"

# Get screen resolution and set default to 80% of screen size
user32 = ctypes.windll.user32

DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600

mp_face_mesh = mp.solutions.face_mesh

# Modern UI Colors
UI_BG_COLOR = (30, 30, 35)  # Dark background
UI_ACCENT_COLOR = (100, 200, 255)  # Cyan accent
UI_HOVER_COLOR = (60, 60, 70)  # Hover state
UI_TEXT_COLOR = (240, 240, 245)  # Light text
UI_BORDER_COLOR = (80, 80, 90)  # Subtle borders
UI_SELECTED_COLOR = (255, 180, 80)  # Orange for selected items
UI_REC_COLOR = (255, 70, 70)  # Red for recording


# ============================================================
#              MODERN UI HELPER FUNCTIONS
# ============================================================
def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
    """Draw a rectangle with rounded corners"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw filled rectangles
    if thickness == -1:
        # Main rectangles
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Corner circles
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Draw outline with rounded corners
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

def draw_modern_text(img, text, pos, scale=0.5, color=(255, 255, 255), thickness=1):
    """Draw text with anti-aliasing"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


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
# Search for PNG files in TARGET_FOLDER and all subdirectories
target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "**", "*.png"), recursive=True))
targets = []

# Get all categories (subdirectories)
categories = ["All"]
for entry in os.listdir(TARGET_FOLDER):
    entry_path = os.path.join(TARGET_FOLDER, entry)
    if os.path.isdir(entry_path):
        categories.append(entry)

for f in target_files:
    name = os.path.splitext(os.path.basename(f))[0]
    # JSON should be in the same directory as the image
    json_path = os.path.join(os.path.dirname(f), name + ".json")
    
    # Get category from parent directory
    parent_dir = os.path.basename(os.path.dirname(f))
    category = parent_dir if parent_dir != os.path.basename(TARGET_FOLDER) else "Uncategorized"

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
        "icon": icon,
        "category": category
    })

if len(targets) == 0:
    raise Exception("No targets found in folder.")

print("Loaded targets:", [t["name"] for t in targets])
print("Categories:", categories)

active_target_index = 0
active_target = targets[active_target_index]
selected_category = "All"
filtered_targets = targets.copy()
dropdown_open = False
icon_scroll_offset = 0  # For sliding icon bar
alpha_value = 50  # Alpha blending value (0-100)
slider_dragging = False  # Track if slider is being dragged

# Build Delaunay triangulation ONCE using first target
tri = Delaunay(active_target["pts"])
triangles = tri.simplices


# ============================================================
#                 OPEN WEBCAM + MESH
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Fix dark camera image
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Enable auto-exposure
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Adjust brightness (0-1)
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)  # Adjust contrast
cap.set(cv2.CAP_PROP_GAIN, 0)  # Auto gain

# PERFORMANCE: Reduce to 1 face and lower confidence for faster processing
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=False,
                                  min_detection_confidence=0.3,
                                  min_tracking_confidence=0.3)

# Create resizable window with default size
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DEFAULT_WIDTH, DEFAULT_HEIGHT)


# ============================================================
#                  MOUSE CLICK HANDLER
# ============================================================
def mouse_event(event, x, y, flags, param):
    global active_target_index, active_target, dropdown_open, selected_category, filtered_targets, icon_scroll_offset, alpha_value, slider_dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        # Get current window dimensions for responsive calculations
        window_rect = cv2.getWindowImageRect(WINDOW_NAME)
        curr_w = window_rect[2] if window_rect[2] > 0 else DEFAULT_WIDTH
        curr_h = window_rect[3] if window_rect[3] > 0 else DEFAULT_HEIGHT
        
        # Calculate slider bounds dynamically
        slider_start_x = int(curr_w * 0.31)
        slider_end_x = int(curr_w * 0.94)
        top_panel_height = int(curr_h * 0.125)
        slider_y_top = int(top_panel_height * 0.25)
        slider_y_bottom = int(top_panel_height * 0.75)
        slider_track_start = slider_start_x + int(curr_w * 0.016)
        slider_track_end = slider_end_x - int(curr_w * 0.016)
        
        # Check custom alpha slider
        if slider_start_x <= x <= slider_end_x and slider_y_top <= y <= slider_y_bottom:
            slider_dragging = True
            # Calculate alpha from click position
            slider_track_width = slider_track_end - slider_track_start
            alpha_value = int(((x - slider_track_start) / slider_track_width) * 100)
            alpha_value = max(0, min(100, alpha_value))
            return
        # Check dropdown toggle button (top panel)
        if 10 <= x <= 180 and 10 <= y <= 45:
            dropdown_open = not dropdown_open
            return
        
        # Check dropdown menu items (if open)
        if dropdown_open and 10 <= x <= 180:
            for i, cat in enumerate(categories):
                y_start = 55 + i * 35
                y_end = y_start + 30
                if y_start <= y <= y_end:
                    selected_category = cat
                    dropdown_open = False
                    
                    # Filter targets based on selected category
                    if selected_category == "All":
                        filtered_targets = targets.copy()
                    else:
                        filtered_targets = [t for t in targets if t["category"] == selected_category]
                    
                    # Reset active target and scroll to first
                    if len(filtered_targets) > 0:
                        active_target_index = 0
                        active_target = filtered_targets[active_target_index]
                        icon_scroll_offset = 0
                        print(f"Category: {selected_category}, Targets: {[t['name'] for t in filtered_targets]}")
                    return
        
        # Check left/right scroll arrows for icon bar
        icon_bar_height = int(curr_h * 0.1875)
        bar_y = curr_h - icon_bar_height
        icon_size = int(min(curr_h * 0.146, curr_w * 0.109))
        icon_spacing = int(icon_size * 1.14)
        arrow_size = int(curr_w * 0.055)
        available_width = curr_w - (arrow_size * 2) - 20
        max_visible_icons = max(1, int(available_width / icon_spacing))
        
        if bar_y <= y <= curr_h:
            arrow_y_center = bar_y + icon_bar_height // 2
            # Left arrow
            if 5 <= x <= 5 + arrow_size and icon_scroll_offset > 0:
                icon_scroll_offset -= 1
                return
            # Right arrow
            max_scroll = max(0, len(filtered_targets) - max_visible_icons)
            if curr_w - 5 - arrow_size <= x <= curr_w - 5 and icon_scroll_offset < max_scroll:
                icon_scroll_offset += 1
                return
        
        # Check icon bar
        if y >= bar_y:
            visible_start = icon_scroll_offset
            visible_end = min(visible_start + max_visible_icons, len(filtered_targets))
            x0 = arrow_size + 15
            
            for i in range(visible_start, visible_end):
                display_index = i - visible_start
                icon_x = x0 + display_index * icon_spacing
                icon_x1 = icon_x
                icon_x2 = icon_x + icon_size
                if icon_x1 <= x <= icon_x2:
                    active_target_index = i
                    active_target = filtered_targets[i]
                    print("Switched to:", active_target["name"])
                    return
        
        # Click outside dropdown closes it
        if dropdown_open:
            dropdown_open = False
    
    elif event == cv2.EVENT_MOUSEMOVE:
        # Update slider while dragging
        if slider_dragging:
            window_rect = cv2.getWindowImageRect(WINDOW_NAME)
            curr_w = window_rect[2] if window_rect[2] > 0 else DEFAULT_WIDTH
            slider_start_x = int(curr_w * 0.31)
            slider_end_x = int(curr_w * 0.94)
            slider_track_start = slider_start_x + int(curr_w * 0.016)
            slider_track_end = slider_end_x - int(curr_w * 0.016)
            
            if slider_start_x <= x <= slider_end_x:
                slider_track_width = slider_track_end - slider_track_start
                alpha_value = int(((x - slider_track_start) / slider_track_width) * 100)
                alpha_value = max(0, min(100, alpha_value))
    
    elif event == cv2.EVENT_LBUTTONUP:
        slider_dragging = False

cv2.setMouseCallback(WINDOW_NAME, mouse_event)

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

    # Get current window size dynamically
    window_rect = cv2.getWindowImageRect(WINDOW_NAME)
    FRAME_W = window_rect[2] if window_rect[2] > 0 else DEFAULT_WIDTH
    FRAME_H = window_rect[3] if window_rect[3] > 0 else DEFAULT_HEIGHT

    # PERFORMANCE: Process at lower resolution for face detection
    PROCESS_W = 320
    PROCESS_H = 240
    frame_small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
    rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    # Resize frame to display size
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    display = frame.copy()

    if results.multi_face_landmarks:
        # Process each detected face
        for face_lms in results.multi_face_landmarks:

            # PERFORMANCE: Scale landmarks from processed resolution to display resolution
            scale_x = FRAME_W / PROCESS_W
            scale_y = FRAME_H / PROCESS_H
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
                
                # PERFORMANCE: Expand teeth mask to ensure full teeth coverage
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
                
                # PERFORMANCE: Expand eyes mask to ensure full coverage (reduced iterations)
                kernel = np.ones((3,3), np.uint8)
                eyes_mask = cv2.dilate(eyes_mask, kernel, iterations=1)
                
                # Remove eyes area from face mask (so real eyes always show)
                face_mask = cv2.subtract(face_mask, eyes_mask)
            
            # PERFORMANCE: Blur mask edges for smooth blending (reduced kernel size)
            face_mask = cv2.GaussianBlur(face_mask, (7, 7), 5)
            face_mask_3ch = cv2.merge([face_mask, face_mask, face_mask]) / 255.0
            
            # Alpha blend
            alpha = alpha_value / 100
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
                
                # PERFORMANCE: Expand to ensure full coverage (reduced iterations)
                kernel = np.ones((3,3), np.uint8)
                final_exclusion_mask = cv2.dilate(final_exclusion_mask, kernel, iterations=1)
                
                # Convert to 3 channels
                final_exclusion_mask_3ch = cv2.merge([final_exclusion_mask, final_exclusion_mask, final_exclusion_mask]) / 255.0
                
                # Replace with original frame in these areas (100% original, 0% morphed)
                display = (frame * final_exclusion_mask_3ch + display * (1 - final_exclusion_mask_3ch)).astype(np.uint8)

    # Write frame BEFORE adding UI elements (so they don't appear in the video)
    if is_recording and video_writer is not None:
        video_writer.write(display)
    
    # ------------ DRAW DARK MODE TOP PANEL (after recording frame) ------------
    # Calculate responsive dimensions
    top_panel_height = int(FRAME_H * 0.125)  # 12.5% of height
    
    # Draw dark background for entire top area
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_W, top_panel_height), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.95, display, 0.05, 0, display)
    
    # ------------ DRAW CUSTOM ALPHA SLIDER ------------
    # Calculate slider dimensions based on window size
    slider_start_x = int(FRAME_W * 0.31)
    slider_end_x = int(FRAME_W * 0.94)
    slider_y_top = int(top_panel_height * 0.25)
    slider_y_bottom = int(top_panel_height * 0.75)
    slider_track_height = int((slider_y_bottom - slider_y_top) * 0.33)
    
    # Slider background
    draw_rounded_rect(display, (slider_start_x, slider_y_top), (slider_end_x, slider_y_bottom), UI_BG_COLOR, -1, 8)
    
    # Slider track
    track_y_center = (slider_y_top + slider_y_bottom) // 2
    track_y_top = track_y_center - slider_track_height // 2
    track_y_bottom = track_y_center + slider_track_height // 2
    slider_track_start = slider_start_x + int(FRAME_W * 0.016)
    slider_track_end = slider_end_x - int(FRAME_W * 0.016)
    
    draw_rounded_rect(display, (slider_track_start, track_y_top), (slider_track_end, track_y_bottom), UI_BORDER_COLOR, -1, 5)
    
    # Slider filled portion (shows current value)
    slider_track_width = slider_track_end - slider_track_start
    slider_fill_end = slider_track_start + int((alpha_value / 100) * slider_track_width)
    if slider_fill_end > slider_track_start:
        draw_rounded_rect(display, (slider_track_start, track_y_top), (slider_fill_end, track_y_bottom), UI_ACCENT_COLOR, -1, 5)
    
    # Slider handle (knob)
    handle_x = slider_track_start + int((alpha_value / 100) * slider_track_width)
    handle_radius = int(FRAME_H * 0.021)
    cv2.circle(display, (handle_x, track_y_center), handle_radius, UI_ACCENT_COLOR, -1, cv2.LINE_AA)
    cv2.circle(display, (handle_x, track_y_center), int(handle_radius * 0.8), UI_BG_COLOR, -1, cv2.LINE_AA)
    cv2.circle(display, (handle_x, track_y_center), handle_radius, UI_ACCENT_COLOR, 2, cv2.LINE_AA)
    
    # Slider label
    label_scale = max(0.3, min(0.5, FRAME_W / 1280))
    draw_modern_text(display, f"Blend: {alpha_value}%", (slider_track_start + 5, slider_y_bottom + int(FRAME_H * 0.013)), label_scale, UI_TEXT_COLOR, 1)
    
    # ------------ DRAW MODERN DROPDOWN MENU (after recording frame) ------------
    # Draw dropdown button with modern style
    draw_rounded_rect(display, (10, 10), (180, 45), UI_BG_COLOR, -1, 8)
    draw_rounded_rect(display, (10, 10), (180, 45), UI_ACCENT_COLOR, 2, 8)
    
    # Category text
    draw_modern_text(display, selected_category, (20, 30), 0.55, UI_TEXT_COLOR, 1)
    
    # Draw dropdown arrow with better style
    arrow_x = 160
    arrow_y = 24 if not dropdown_open else 27
    pts = np.array([[arrow_x - 5, arrow_y - 3], [arrow_x, arrow_y + 2], [arrow_x + 5, arrow_y - 3]], np.int32)
    cv2.fillPoly(display, [pts], UI_ACCENT_COLOR, cv2.LINE_AA)
    
    # Draw dropdown menu (if open) with modern style
    if dropdown_open:
        menu_height = len(categories) * 35 + 10
        # Draw shadow
        shadow_overlay = display.copy()
        draw_rounded_rect(shadow_overlay, (13, 53), (183, 53 + menu_height), (0, 0, 0), -1, 8)
        cv2.addWeighted(shadow_overlay, 0.3, display, 0.7, 0, display)
        
        # Draw menu background
        draw_rounded_rect(display, (10, 50), (180, 50 + menu_height), UI_BG_COLOR, -1, 8)
        draw_rounded_rect(display, (10, 50), (180, 50 + menu_height), UI_BORDER_COLOR, 1, 8)
        
        for i, cat in enumerate(categories):
            y_pos = 55 + i * 35
            # Highlight selected or hovered category
            if cat == selected_category:
                draw_rounded_rect(display, (15, y_pos), (175, y_pos + 30), UI_ACCENT_COLOR, -1, 6)
                text_color = (20, 20, 30)
            else:
                text_color = UI_TEXT_COLOR
            
            draw_modern_text(display, cat, (25, y_pos + 20), 0.5, text_color, 1)
    
    # ------------ DRAW MODERN ICON BAR WITH SLIDING (after recording frame) ------------
    # Calculate responsive icon bar dimensions
    icon_bar_height = int(FRAME_H * 0.1875)  # 18.75% of height
    icon_y = FRAME_H - icon_bar_height
    icon_size = int(min(FRAME_H * 0.146, FRAME_W * 0.109))  # Scale icons based on window size
    icon_spacing = int(icon_size * 1.14)
    arrow_size = int(FRAME_W * 0.055)
    
    # Draw semi-transparent background bar
    overlay = display.copy()
    draw_rounded_rect(overlay, (0, icon_y - 5), (FRAME_W, FRAME_H), (20, 20, 25), -1, 10)
    cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
    
    # Calculate how many icons can fit
    available_width = FRAME_W - (arrow_size * 2) - 20
    max_visible_icons = max(1, int(available_width / icon_spacing))
    
    # Draw modern left arrow (if not at start)
    if icon_scroll_offset > 0:
        arrow_y_center = icon_y + icon_bar_height // 2
        draw_rounded_rect(display, (5, arrow_y_center - arrow_size//2), (5 + arrow_size, arrow_y_center + arrow_size//2), UI_BG_COLOR, -1, 8)
        draw_rounded_rect(display, (5, arrow_y_center - arrow_size//2), (5 + arrow_size, arrow_y_center + arrow_size//2), UI_ACCENT_COLOR, 2, 8)
        # Draw left chevron
        arrow_offset = int(arrow_size * 0.25)
        pts = np.array([[5 + arrow_size - arrow_offset, arrow_y_center], [5 + arrow_offset, arrow_y_center], [5 + arrow_offset + 4, arrow_y_center - 7]], np.int32)
        cv2.fillPoly(display, [pts], UI_ACCENT_COLOR, cv2.LINE_AA)
        pts = np.array([[5 + arrow_size - arrow_offset, arrow_y_center], [5 + arrow_offset, arrow_y_center], [5 + arrow_offset + 4, arrow_y_center + 7]], np.int32)
        cv2.fillPoly(display, [pts], UI_ACCENT_COLOR, cv2.LINE_AA)
    
    # Draw modern right arrow (if more icons to show)
    max_scroll = max(0, len(filtered_targets) - max_visible_icons)
    if icon_scroll_offset < max_scroll:
        arrow_y_center = icon_y + icon_bar_height // 2
        draw_rounded_rect(display, (FRAME_W - 5 - arrow_size, arrow_y_center - arrow_size//2), (FRAME_W - 5, arrow_y_center + arrow_size//2), UI_BG_COLOR, -1, 8)
        draw_rounded_rect(display, (FRAME_W - 5 - arrow_size, arrow_y_center - arrow_size//2), (FRAME_W - 5, arrow_y_center + arrow_size//2), UI_ACCENT_COLOR, 2, 8)
        # Draw right chevron
        arrow_offset = int(arrow_size * 0.25)
        pts = np.array([[FRAME_W - 5 - arrow_size + arrow_offset, arrow_y_center], [FRAME_W - 5 - arrow_offset, arrow_y_center], [FRAME_W - 5 - arrow_offset - 4, arrow_y_center - 7]], np.int32)
        cv2.fillPoly(display, [pts], UI_ACCENT_COLOR, cv2.LINE_AA)
        pts = np.array([[FRAME_W - 5 - arrow_size + arrow_offset, arrow_y_center], [FRAME_W - 5 - arrow_offset, arrow_y_center], [FRAME_W - 5 - arrow_offset - 4, arrow_y_center + 7]], np.int32)
        cv2.fillPoly(display, [pts], UI_ACCENT_COLOR, cv2.LINE_AA)
    
    # Draw visible icons with modern style
    visible_start = icon_scroll_offset
    visible_end = min(visible_start + max_visible_icons, len(filtered_targets))
    x0 = arrow_size + 15
    
    for i in range(visible_start, visible_end):
        t = filtered_targets[i]
        
        center_x = x0 + icon_size // 2
        center_y = icon_y + icon_bar_height // 2
        
        # Draw selection ring with glow effect
        if i == active_target_index:
            # Outer glow
            cv2.circle(display, (center_x, center_y), int(icon_size * 0.57), UI_SELECTED_COLOR, 2, cv2.LINE_AA)
            cv2.circle(display, (center_x, center_y), int(icon_size * 0.54), UI_SELECTED_COLOR, 1, cv2.LINE_AA)
        else:
            # Subtle border for unselected
            cv2.circle(display, (center_x, center_y), int(icon_size * 0.53), UI_BORDER_COLOR, 1, cv2.LINE_AA)
        
        # Create circular mask for icon
        icon_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cv2.circle(icon_mask, (center_x, center_y), int(icon_size * 0.5), 255, -1)
        
        # Apply icon with circular mask
        icon_y1 = max(0, center_y - icon_size // 2)
        icon_y2 = min(FRAME_H, center_y + icon_size // 2)
        icon_x1 = max(0, x0)
        icon_x2 = min(FRAME_W, x0 + icon_size)
        
        if icon_y2 > icon_y1 and icon_x2 > icon_x1:
            icon_region = display[icon_y1:icon_y2, icon_x1:icon_x2]
            icon_mask_region = icon_mask[icon_y1:icon_y2, icon_x1:icon_x2]
            icon_mask_3ch = cv2.merge([icon_mask_region, icon_mask_region, icon_mask_region]) / 255.0
            
            # Resize target icon to match current icon size
            resized_icon = cv2.resize(t["icon"], (icon_size, icon_size))
            icon_src = resized_icon[0:icon_region.shape[0], 0:icon_region.shape[1]]
            
            if icon_src.shape == icon_region.shape:
                icon_region[:] = (icon_src * icon_mask_3ch[0:icon_src.shape[0], 0:icon_src.shape[1]] + 
                                icon_region * (1 - icon_mask_3ch[0:icon_src.shape[0], 0:icon_src.shape[1]])).astype(np.uint8)
        
        x0 += icon_spacing
    
    # Add modern recording indicator (only shows in window)
    if is_recording:
        # Calculate responsive recording indicator dimensions
        rec_width = int(FRAME_W * 0.117)
        rec_height = int(top_panel_height * 0.583)
        rec_x1 = FRAME_W - rec_width - int(FRAME_W * 0.023)
        rec_y1 = int(top_panel_height * 0.25)
        rec_x2 = FRAME_W - int(FRAME_W * 0.023)
        rec_y2 = rec_y1 + rec_height
        
        # Animated pulse effect
        base_pulse = int(FRAME_H * 0.025)
        pulse_size = base_pulse + int(base_pulse * 0.25 * np.sin(time.time() * 3))
        draw_rounded_rect(display, (rec_x1, rec_y1), (rec_x2, rec_y2), UI_BG_COLOR, -1, 8)
        
        circle_x = rec_x1 + int(rec_width * 0.27)
        circle_y = (rec_y1 + rec_y2) // 2
        cv2.circle(display, (circle_x, circle_y), pulse_size, UI_REC_COLOR, -1, cv2.LINE_AA)
        
        text_scale = max(0.4, min(0.7, FRAME_W / 1000))
        text_x = circle_x + pulse_size + int(FRAME_W * 0.008)
        draw_modern_text(display, "REC", (text_x, circle_y + int(FRAME_H * 0.008)), text_scale, UI_REC_COLOR, 2)
    
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
