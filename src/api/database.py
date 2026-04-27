from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import os

os.makedirs("data/db", exist_ok=True)
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/db/anpr.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    image_path = Column(String, index=True) 
    
    predicted_plate = Column(String, index=True)
    confidence_score = Column(Float)  
    format_score = Column(Float)      
    
    is_reviewed = Column(Boolean, default=False)
    corrected_plate = Column(String, nullable=True)

class VehicleRecord(Base):
    __tablename__ = "vehicles"

    plate_number = Column(String, primary_key=True, index=True)
    vehicle_make = Column(String)
    vehicle_model = Column(String)
    vehicle_year = Column(Integer, nullable=True)
    vehicle_color = Column(String)
    vehicle_type = Column(String)
    owner_name = Column(String)
    owner_phone = Column(String, nullable=True)
    owner_email = Column(String, nullable=True)
    owner_address = Column(String, nullable=True)
    registered_rto_state = Column(String)
    registered_rto_code = Column(String)
    
    chassis_number = Column(String, nullable=True)
    engine_number = Column(String, nullable=True)
    registration_date = Column(String, nullable=True)
    registration_expiry = Column(String, nullable=True)
    insurance_expiry = Column(String, nullable=True)
    
    status = Column(String, default="Clear") # 'Clear', 'Stolen/Missing', 'Recovered'
    police_complaint_id = Column(String, nullable=True)
    missing_date = Column(String, nullable=True)
    recovery_date = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()