import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay
import os
import json
import glob



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

def load_landmarks_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    pts = np.array(data["points"], dtype=np.int32)
    w = int(data["width"])
    h = int(data["height"])
    return pts, w, h
# Load all target images and their landmarks

TARGETS = []

target_files = glob.glob("Targets/**/*.png", recursive=True)

for img_path in target_files:
    name = os.path.splitext(os.path.basename(img_path))[0]
    json_path = img_path.replace(".png", ".json")

    if not os.path.exists(json_path):
        continue

    img = cv2.imread(img_path)
    pts, w, h = load_landmarks_json(json_path)
    img = cv2.resize(img, (w, h))

    TARGETS.append({
        "name": name,
        "img": img,
        "pts": pts
    })

    

if len(TARGETS) == 0:
    raise RuntimeError("No targets found")

ACTIVE_TARGET = TARGETS[0]
TRIANGLES = Delaunay(ACTIVE_TARGET["pts"]).simplices
BLEND_ALPHA = 50  # Default blend value (0-100)

def get_available_targets():
    """Return list of available target names"""
    return [t["name"] for t in TARGETS]

def set_active_target(target_name):
    """Change the active morph target"""
    global ACTIVE_TARGET, TRIANGLES
    for t in TARGETS:
        if t["name"] == target_name:
            ACTIVE_TARGET = t
            TRIANGLES = Delaunay(ACTIVE_TARGET["pts"]).simplices
            return True
    return False

def set_blend_alpha(alpha):
    """Set blend alpha (0-100)"""
    global BLEND_ALPHA
    BLEND_ALPHA = max(0, min(100, alpha))

def warp_triangle(img_src, img_dst, t_src, t_dst):
    try:
        # bounding rectangles
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))
        
        # Skip invalid rectangles
        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            return

        # offset points by top-left corner
        t1_rect = []
        t2_rect = []
        t2_rect_int = []

        for i in range(3):
            t1_rect.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
            t2_rect.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))
            t2_rect_int.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))

        # mask
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

        # crop source
        img1_rect = img_src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        
        # Skip if crop is invalid
        if img1_rect.size == 0:
            return

        # affine transform
        size = (r2[2], r2[3])
        mat = cv2.getAffineTransform(
            np.float32(t1_rect),
            np.float32(t2_rect)
        )

        img2_rect = cv2.warpAffine(
            img1_rect,
            mat,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101
        )

        img2_rect = img2_rect * mask

        # paste to destination with bounds checking
        dst_region = img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
        if dst_region.shape == img2_rect.shape and dst_region.shape == mask.shape:
            img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_region * (1 - mask) + img2_rect
    except Exception:
        # Silently skip problematic triangles
        pass



def process_frame(frame):
    """Original process frame (kept for compatibility)"""
    return process_frame_realtime(frame)

def process_frame_realtime(frame):
    """Process frame with real-time morphing and configurable blend"""
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return frame

    face = results.multi_face_landmarks[0]
    src_pts = np.array(
        [[int(lm.x * w), int(lm.y * h)] for lm in face.landmark],
        dtype=np.int32
    )

    warped = np.zeros_like(frame)

    # Process every other triangle for 50% speed boost
    for i, tri in enumerate(TRIANGLES):
        if i % 2 == 0:  # Skip every other triangle
            t_src = src_pts[tri]
            t_tgt = ACTIVE_TARGET["pts"][tri]
            warp_triangle(ACTIVE_TARGET["img"], warped, t_tgt, t_src)

    # Create face mask to apply alpha only to face region
    face_mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(src_pts)
    cv2.fillConvexPoly(face_mask, hull, 255)
    
    # Skip blur for faster processing
    face_mask_3ch = cv2.merge([face_mask, face_mask, face_mask]) / 255.0
    
    # Apply alpha blending ONLY to face region
    alpha = BLEND_ALPHA / 100.0
    
    # Blend face area with alpha
    face_blend = cv2.addWeighted(frame, 1 - alpha, warped, alpha, 0)
    
    # Combine: blended face inside mask, original frame outside mask
    result = (face_blend * face_mask_3ch + frame * (1 - face_mask_3ch)).astype(np.uint8)
    
    return result






