from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Response
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
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from src.api.ml_service import ANPREngine
from src.api.database import init_db, get_db, DetectionRecord, VehicleRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_engine = None
UPLOAD_DIR = "data/api_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Must be a PNG or JPEG image.")

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
            
        return {"success": True, "image_url": f"/uploads/{unique_filename}", "detections": db_records}

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
async def process_video_compat(file: UploadFile = File(...)):
    """Compatibility endpoint for legacy frontend video requests."""
    return {
        "job_id": f"video-{uuid.uuid4().hex[:8]}",
        "status": "error",
        "input_type": "video",
        "total_detections": 0,
        "processing_time": 0.0,
        "detections": [],
        "error": "Video processing is not available in this backend build yet.",
        "progress": 100,
        "stage": "Unsupported",
        "alert_count": 0,
        "review_count": 0,
    }


@app.get("/api/results/{job_id}")
async def get_job_results_compat(job_id: str):
    """Compatibility endpoint for legacy frontend polling requests."""
    return {
        "job_id": job_id,
        "status": "error",
        "input_type": "video",
        "total_detections": 0,
        "processing_time": 0.0,
        "detections": [],
        "error": "Job polling is not used in the current synchronous image pipeline.",
        "progress": 100,
        "stage": "Completed",
        "alert_count": 0,
        "review_count": 0,
    }

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)