from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
import cv2
import numpy as np
import json

from morphEngine import process_frame_realtime, get_available_targets, set_active_target, set_blend_alpha

app = FastAPI()

# Enable CORS for web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    """Serve the HTML page"""
    return FileResponse("index.html")

@app.get("/ping")
def ping():
    return {"message": "Python server is alive"}

@app.get("/targets")
def get_targets():
    """Get list of available morph targets"""
    targets = get_available_targets()
    return {"targets": targets}

@app.post("/set-target")
async def set_target(data: dict):
    """Set the active morph target"""
    target_name = data.get("target")
    success = set_active_target(target_name)
    return {"success": success, "active_target": target_name}

@app.post("/set-blend")
async def set_blend(data: dict):
    """Set the blend alpha value (0-100)"""
    alpha = data.get("alpha", 50)
    set_blend_alpha(alpha)
    return {"success": True, "alpha": alpha}

@app.post("/morph")
async def morph(file: UploadFile = File(...)):
    """Original single image morph endpoint"""
    # 1) read image from web
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    # 2) run YOUR morph code
    out = process_frame_realtime(img)

    # 3) send image back
    _, buf = cv2.imencode(".jpg", out)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.post("/morph-realtime")
async def morph_realtime(file: UploadFile = File(...)):
    """Real-time video frame morphing endpoint"""
    # 1) Read frame from video stream
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    
    if frame is None:
        return Response(content=b"", media_type="image/jpeg", status_code=400)

    # 2) Process frame with face morphing
    morphed_frame = process_frame_realtime(frame)

    # 3) Encode and return
    _, buf = cv2.imencode(".jpg", morphed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return Response(content=buf.tobytes(), media_type="image/jpeg")
