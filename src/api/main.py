from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging
import uuid
import os
import shutil
import csv
import io
import re
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import mimetypes
import cv2
import numpy as np
from pathlib import Path
import json

from src.api.ml_service import ANPREngine
from src.api.database import init_db, get_db, DetectionRecord, VehicleRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_engine = None
UPLOAD_DIR = "data/api_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Video Processing Job State (in-memory, use Redis in production)
JOB_STATE = {}  # {job_id: {status, progress, detections, error, ...}}
VIDEOS_DIR = "data/api_videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Security Configuration
MAX_UPLOAD_SIZE_MB = 10  # 10 MB max per image
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# CORS Configuration (restrict to known origins in production)
ALLOWED_ORIGINS = [
    "http://localhost:5173",           # Local dev
    "http://localhost:3000",           # Alt local dev
    "http://127.0.0.1:5173",           # Localhost variant
    "http://172.27.197.240:5173",      # WSL IP (local dev)
    # Add production domain here: "https://yourdomain.com"
]

CORS_CONFIG = {
    "allow_origins": ALLOWED_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE"],
    "allow_headers": ["*"],
}


def validate_image_upload(file: UploadFile) -> None:
    """
    Validate image upload before processing.
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If file fails validation
    """
    # Check file size
    if file.size and file.size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Check MIME type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    logger.info(f"Upload validation passed for {file.filename} ({file.content_type})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_engine
    logger.info("Initializing Database...")
    init_db()
    
    logger.info("Starting up API server...")
    YOLO_PATH = "models/checkpoints/plate_detector_v1.pt"
    ml_engine = ANPREngine(yolo_weights=YOLO_PATH)
    yield
    logger.info("Shutting down API server...")
    ml_engine = None

app = FastAPI(title="ANPR API", version="1.0", lifespan=lifespan)

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    **CORS_CONFIG
)

# Serve the cropped plate images to the frontend
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def serialize_vehicle(record: VehicleRecord) -> dict:
    return {
        "plate_number": record.plate_number,
        "vehicle_make": record.vehicle_make,
        "vehicle_model": record.vehicle_model,
        "vehicle_year": record.vehicle_year,
        "vehicle_color": record.vehicle_color,
        "vehicle_type": record.vehicle_type,
        "owner_name": record.owner_name,
        "owner_phone": record.owner_phone,
        "owner_email": record.owner_email,
        "owner_address": record.owner_address,
        "registered_rto_state": record.registered_rto_state,
        "registered_rto_code": record.registered_rto_code,
        "chassis_number": record.chassis_number,
        "engine_number": record.engine_number,
        "registration_date": record.registration_date,
        "registration_expiry": record.registration_expiry,
        "insurance_expiry": record.insurance_expiry,
        "status": record.status,
        "police_complaint_id": record.police_complaint_id,
        "missing_date": record.missing_date,
        "recovery_date": record.recovery_date,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }

@app.get("/")
async def root():
    return {"status": "online", "message": "ANPR API is running."}

@app.post("/api/v1/detect/image")
async def detect_license_plate(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Detect license plates in uploaded image.
    
    Security: Validates file size and type before processing.
    """
    # Validate upload
    validate_image_upload(file)
    
    try:
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        result = ml_engine.predict(image_bytes)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error"))
            
        db_records = []
        for det in result["detections"]:
            plate_text = det["plate_text"]
            
            # 1. Save to Detection History
            new_record = DetectionRecord(
                image_path=f"/uploads/{unique_filename}",
                predicted_plate=plate_text,
                confidence_score=det["confidence"],
                format_score=det["format_score"]
            )
            db.add(new_record)
            db.commit()
            db.refresh(new_record)
            
            # 2. CROSS-REFERENCE WITH VEHICLE REGISTRY
            registry_match = None
            vehicle = db.query(VehicleRecord).filter(VehicleRecord.plate_number == plate_text).first()
            
            if vehicle:
                registry_match = {
                    "plate_number": vehicle.plate_number,
                    "status": vehicle.status,
                    "vehicle_make": vehicle.vehicle_make,
                    "vehicle_model": vehicle.vehicle_model,
                    "owner_name": vehicle.owner_name,
                    "registered_rto_state": vehicle.registered_rto_state,
                    "registered_rto_code": vehicle.registered_rto_code,
                    "police_complaint_id": vehicle.police_complaint_id
                }
            
            # 3. Append to the React response payload
            db_records.append({
                "db_id": new_record.id, 
                "registry_match": registry_match,
                **det
            })
            
        logger.info(f"Detection completed: {len(db_records)} plates found")
        return {"success": True, "image_url": f"/uploads/{unique_filename}", "detections": db_records}

    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/api/v1/detect/correct/{record_id}")
async def correct_plate(record_id: int, corrected_text: str, db: Session = Depends(get_db)):
    record = db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Detection record not found.")
        
    record.corrected_plate = corrected_text.upper().replace(" ", "")
    record.is_reviewed = True
    db.commit()
    return {"success": True, "message": "Correction saved to database for active learning."}

@app.get("/api/vehicles")
async def list_vehicles(
    status: str = None,
    search: str = None,
    vehicle_type: str = None,
    state_code: str = None,
    db: Session = Depends(get_db),
):
    """Powers the Vehicle Registry dashboard in React."""
    query = db.query(VehicleRecord)
    if status:
        query = query.filter(VehicleRecord.status == status)
    if search:
        query = query.filter(VehicleRecord.plate_number.contains(search.upper()))
    if vehicle_type:
        query = query.filter(VehicleRecord.vehicle_type == vehicle_type)
    if state_code:
        query = query.filter(VehicleRecord.registered_rto_code == state_code)
        
    records = query.all()
    return [serialize_vehicle(r) for r in records]

# Pydantic Schemas for React Payloads
class VehicleCreateUpdate(BaseModel):
    plate_number: Optional[str] = None
    vehicle_make: str
    vehicle_model: str
    vehicle_year: Optional[int] = None
    vehicle_color: str
    vehicle_type: Optional[str] = "Car"
    owner_name: str
    owner_phone: Optional[str] = None
    owner_email: Optional[str] = None
    owner_address: Optional[str] = None
    registered_rto_state: str
    registered_rto_code: str
    chassis_number: Optional[str] = None
    engine_number: Optional[str] = None
    registration_date: Optional[str] = None
    registration_expiry: Optional[str] = None
    insurance_expiry: Optional[str] = None

class ComplaintRequest(BaseModel):
    complaint_id: str
    reported_by: Optional[str] = None
    reporting_station: Optional[str] = None
    theft_location: Optional[str] = None


class VehicleUpdate(BaseModel):
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_year: Optional[int] = None
    vehicle_color: Optional[str] = None
    vehicle_type: Optional[str] = None
    owner_name: Optional[str] = None
    owner_phone: Optional[str] = None
    owner_email: Optional[str] = None
    owner_address: Optional[str] = None
    registered_rto_state: Optional[str] = None
    registered_rto_code: Optional[str] = None
    chassis_number: Optional[str] = None
    engine_number: Optional[str] = None
    registration_date: Optional[str] = None
    registration_expiry: Optional[str] = None
    insurance_expiry: Optional[str] = None

# --- NEW VEHICLE REGISTRY ENDPOINTS ---

@app.post("/api/vehicles")
async def create_vehicle(payload: VehicleCreateUpdate, db: Session = Depends(get_db)):
    """Adds a new vehicle from the React modal."""
    if not payload.plate_number:
        raise HTTPException(status_code=400, detail="Plate number is required")
        
    existing = db.query(VehicleRecord).filter(VehicleRecord.plate_number == payload.plate_number).first()
    if existing:
        raise HTTPException(status_code=400, detail="Vehicle with this plate already exists")
        
    new_vehicle = VehicleRecord(**payload.dict())
    db.add(new_vehicle)
    db.commit()
    db.refresh(new_vehicle)
    return {"success": True, "message": "Vehicle added", "vehicle": serialize_vehicle(new_vehicle)}


@app.get("/api/vehicles/record/{plate}")
async def get_vehicle(plate: str, db: Session = Depends(get_db)):
    vehicle = db.query(VehicleRecord).filter(VehicleRecord.plate_number == plate.upper()).first()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return serialize_vehicle(vehicle)


@app.put("/api/vehicles/record/{plate}")
async def update_vehicle(plate: str, payload: VehicleUpdate, db: Session = Depends(get_db)):
    vehicle = db.query(VehicleRecord).filter(VehicleRecord.plate_number == plate.upper()).first()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    updates = payload.dict(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No update fields provided")

    for key, value in updates.items():
        setattr(vehicle, key, value)

    vehicle.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(vehicle)
    return {"success": True, "message": "Vehicle updated", "vehicle": serialize_vehicle(vehicle)}


@app.get("/api/vehicles/export")
async def export_vehicles(status: str = None, search: str = None, vehicle_type: str = None, state_code: str = None, db: Session = Depends(get_db)):
    query = db.query(VehicleRecord)
    if status:
        query = query.filter(VehicleRecord.status == status)
    if search:
        query = query.filter(VehicleRecord.plate_number.contains(search.upper()))
    if vehicle_type:
        query = query.filter(VehicleRecord.vehicle_type == vehicle_type)
    if state_code:
        query = query.filter(VehicleRecord.registered_rto_code == state_code)

    records = query.all()
    rows = [serialize_vehicle(r) for r in records]
    output = io.StringIO()
    fieldnames = list(rows[0].keys()) if rows else [
        "plate_number", "vehicle_make", "vehicle_model", "vehicle_year", "vehicle_color", "vehicle_type",
        "owner_name", "owner_phone", "owner_email", "owner_address", "registered_rto_state", "registered_rto_code",
        "chassis_number", "engine_number", "registration_date", "registration_expiry", "insurance_expiry", "status",
        "police_complaint_id", "missing_date", "recovery_date", "created_at", "updated_at"
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vehicle-registry.csv"},
    )


@app.post("/api/vehicles/import")
async def import_vehicles(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))

    imported = 0
    updated = 0
    errors = []

    for idx, row in enumerate(reader, start=2):
        plate = (row.get("plate_number") or "").strip().upper()
        if not plate:
            errors.append(f"line {idx}: missing plate_number")
            continue

        payload = {
            "plate_number": plate,
            "vehicle_make": (row.get("vehicle_make") or "Unknown").strip(),
            "vehicle_model": (row.get("vehicle_model") or "Unknown").strip(),
            "vehicle_year": int(row["vehicle_year"]) if row.get("vehicle_year") else None,
            "vehicle_color": (row.get("vehicle_color") or "Unknown").strip(),
            "vehicle_type": (row.get("vehicle_type") or "Car").strip(),
            "owner_name": (row.get("owner_name") or "Unknown").strip(),
            "owner_phone": (row.get("owner_phone") or None),
            "owner_email": (row.get("owner_email") or None),
            "owner_address": (row.get("owner_address") or None),
            "registered_rto_state": (row.get("registered_rto_state") or "NA").strip(),
            "registered_rto_code": (row.get("registered_rto_code") or "00").strip(),
            "chassis_number": (row.get("chassis_number") or None),
            "engine_number": (row.get("engine_number") or None),
            "registration_date": (row.get("registration_date") or None),
            "registration_expiry": (row.get("registration_expiry") or None),
            "insurance_expiry": (row.get("insurance_expiry") or None),
        }

        existing = db.query(VehicleRecord).filter(VehicleRecord.plate_number == plate).first()
        if existing:
            for key, value in payload.items():
                if key != "plate_number":
                    setattr(existing, key, value)
            existing.updated_at = datetime.now(timezone.utc)
            updated += 1
        else:
            db.add(VehicleRecord(**payload))
            imported += 1

    db.commit()
    return {"success": True, "imported": imported, "updated": updated, "errors": errors}

@app.delete("/api/vehicles/record/{plate}")
async def delete_vehicle(plate: str, db: Session = Depends(get_db)):
    """Deletes a vehicle."""
    vehicle = db.query(VehicleRecord).filter(VehicleRecord.plate_number == plate.upper()).first()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
        
    db.delete(vehicle)
    db.commit()
    return {"success": True, "message": "Vehicle deleted", "vehicle": None}

@app.post("/api/vehicles/record/{plate}/file-complaint")
async def file_complaint(plate: str, payload: ComplaintRequest, db: Session = Depends(get_db)):
    """Updates vehicle status to Stolen/Missing and adds FIR details."""
    vehicle = db.query(VehicleRecord).filter(VehicleRecord.plate_number == plate.upper()).first()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
        
    vehicle.status = "Stolen/Missing"
    vehicle.police_complaint_id = payload.complaint_id
    vehicle.missing_date = datetime.now(timezone.utc).isoformat()
    vehicle.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(vehicle)
    return {"success": True, "message": "Complaint filed successfully", "vehicle": serialize_vehicle(vehicle)}

@app.post("/api/vehicles/record/{plate}/mark-recovered")
async def mark_recovered(plate: str, db: Session = Depends(get_db)):
    """Updates vehicle status to Recovered."""
    vehicle = db.query(VehicleRecord).filter(VehicleRecord.plate_number == plate.upper()).first()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
        
    vehicle.status = "Recovered"
    vehicle.recovery_date = datetime.now(timezone.utc).isoformat()
    vehicle.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(vehicle)
    return {"success": True, "message": "Vehicle marked as recovered", "vehicle": serialize_vehicle(vehicle)}


@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Process video file for license plate detection.
    
    Returns immediately with a job_id for polling progress.
    Use GET /api/results/{job_id} to check progress.
    """
    # Validate video file
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Must be a video file (mp4, avi, mov, mkv)")
    
    # Save video
    job_id = f"video-{uuid.uuid4().hex[:12]}"
    file_extension = os.path.splitext(file.filename)[1]
    video_filename = f"{job_id}{file_extension}"
    video_path = os.path.join(VIDEOS_DIR, video_filename)
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving video: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video file")
    
    # Initialize job state
    JOB_STATE[job_id] = {
        "status": "processing",
        "progress": 0,
        "total_frames": 0,
        "processed_frames": 0,
        "detections": [],
        "total_detections": 0,
        "alert_count": 0,
        "review_count": 0,
        "error": None,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "processing_time": 0.0,
        "output_video": None,
    }
    
    # Start background processing
    background_tasks.add_task(process_video_background, job_id, video_path)
    
    logger.info(f"Video processing job started: {job_id}")
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video processing started. Use GET /api/results/{job_id} to check progress.",
        "progress": 0
    }


def process_video_background(job_id: str, video_path: str):
    """Background task to process video."""
    try:
        logger.info(f"Starting video processing for job {job_id}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        JOB_STATE[job_id]["total_frames"] = total_frames
        
        # Prepare output video
        output_filename = f"{job_id}_output.mp4"
        output_path = os.path.join(VIDEOS_DIR, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = datetime.now(timezone.utc)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = ml_engine.detector(frame, verbose=False, conf=0.45)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            
            detections_this_frame = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Dynamic padding
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                pad_x = max(2, int(box_w * 0.05))
                pad_y = max(2, int(box_h * 0.05))
                y1_p, y2_p = max(0, y1 - pad_y), min(height, y2 + pad_y)
                x1_p, x2_p = max(0, x1 - pad_x), min(width, x2 + pad_x)
                
                plate_crop = frame[y1_p:y2_p, x1_p:x2_p]
                if plate_crop.size == 0:
                    continue
                
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
                
                ocr_results = ml_engine.reader.readtext(
                    gray,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.',
                    detail=1,
                    paragraph=False
                )
                
                final_text = ""
                avg_confidence = 0.0
                format_score = 0.0
                
                if ocr_results:
                    # Use same OCR logic as image detection
                    parsed_blocks = []
                    for bbox_ocr, text, conf in ocr_results:
                        if not text or conf <= 0:
                            continue
                        xs = [p[0] for p in bbox_ocr]
                        ys = [p[1] for p in bbox_ocr]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        parsed_blocks.append({
                            "text": text,
                            "conf": float(conf),
                            "height": max(1.0, float(y_max - y_min)),
                            "x_center": float((x_min + x_max) / 2.0),
                            "y_center": float((y_min + y_max) / 2.0),
                        })
                    
                    if parsed_blocks:
                        heights = sorted([b["height"] for b in parsed_blocks])
                        median_h = heights[len(heights) // 2]
                        valid_blocks = [b for b in parsed_blocks if b["height"] >= median_h * 0.5]
                        
                        if valid_blocks:
                            row_tol = max(5.0, median_h * 0.6)
                            by_y = sorted(valid_blocks, key=lambda b: b["y_center"])
                            rows = []
                            
                            for block in by_y:
                                if not rows:
                                    rows.append([block])
                                    continue
                                last_row = rows[-1]
                                row_y = sum(j["y_center"] for j in last_row) / len(last_row)
                                if abs(block["y_center"] - row_y) <= row_tol:
                                    last_row.append(block)
                                else:
                                    rows.append([block])
                            
                            plate_parts = []
                            ordered_blocks = []
                            for row in rows:
                                for block in sorted(row, key=lambda b: b["x_center"]):
                                    plate_parts.append(block["text"])
                                    ordered_blocks.append(block)
                            
                            raw_text = "".join(plate_parts).replace(" ", "")
                            clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                            
                            # Fuzzy state correction
                            if len(clean_text) >= 4:
                                state_code = clean_text[:2]
                                if state_code.isalpha() and state_code not in ml_engine.detector.__dict__.get("VALID_INDIAN_STATES", set()):
                                    import difflib
                                    match = difflib.get_close_matches(state_code, sorted(ml_engine.detector.__dict__.get("VALID_INDIAN_STATES", set())), n=1, cutoff=0.5)
                                    if match:
                                        clean_text = match[0] + clean_text[2:]
                            
                            avg_confidence = sum(b["conf"] for b in ordered_blocks) / len(ordered_blocks)
                            final_text = clean_text
                            format_score = ml_engine._calculate_format_score(final_text)
                
                # Draw on frame
                color = (0, 255, 0) if format_score == 1.0 else (255, 165, 0) if format_score > 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if final_text:
                    (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, final_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                detections_this_frame.append({
                    "plate_text": final_text,
                    "confidence": round(float(avg_confidence), 3),
                    "format_score": format_score,
                    "frame": frame_count
                })
            
            # Write frame
            writer.write(frame)
            
            # Update job state
            JOB_STATE[job_id]["processed_frames"] = frame_count
            JOB_STATE[job_id]["progress"] = int((frame_count / total_frames) * 100)
            JOB_STATE[job_id]["total_detections"] += len(detections_this_frame)
            JOB_STATE[job_id]["detections"].extend(detections_this_frame)
            
            if frame_count % 30 == 0:
                logger.info(f"Job {job_id}: {frame_count}/{total_frames} frames processed")
        
        cap.release()
        writer.release()
        
        # Calculate metrics
        elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        JOB_STATE[job_id].update({
            "status": "completed",
            "progress": 100,
            "processed_frames": frame_count,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "processing_time": round(elapsed_time, 2),
            "output_video": f"/uploads/{output_filename}",
            "alert_count": sum(1 for d in JOB_STATE[job_id]["detections"] if d["format_score"] < 0.8),
            "review_count": sum(1 for d in JOB_STATE[job_id]["detections"] if d["format_score"] == 0.0),
        })
        
        logger.info(f"Video processing completed for job {job_id}: {JOB_STATE[job_id]['total_detections']} detections in {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error processing video {job_id}: {e}")
        JOB_STATE[job_id].update({
            "status": "error",
            "error": str(e),
            "progress": 0,
            "end_time": datetime.now(timezone.utc).isoformat(),
        })


@app.get("/api/results/{job_id}")
async def get_job_results(job_id: str):
    """Get status and results of a video processing job."""
    if job_id not in JOB_STATE:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = JOB_STATE[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "total_frames": job["total_frames"],
        "processed_frames": job["processed_frames"],
        "total_detections": job["total_detections"],
        "alert_count": job["alert_count"],
        "review_count": job["review_count"],
        "processing_time": job["processing_time"],
        "detections": job["detections"][:100],  # Return first 100
        "output_video": job["output_video"],
        "error": job["error"],
        "start_time": job["start_time"],
        "end_time": job["end_time"],
    }

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)