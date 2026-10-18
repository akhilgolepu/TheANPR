# TheANPR: Advanced Number Plate Recognition System

An end-to-end Indian Automatic Number Plate Recognition (ANPR) system powered by YOLOv8 object detection and EasyOCR text recognition, with active learning and production-grade deployment.

## 🎯 Features

- **Real-time Plate Detection**: YOLOv8-based detector achieving 99.47% mAP50
- **High-Accuracy OCR**: EasyOCR with spatial ordering and fuzzy state correction
- **Vehicle Registry**: SQLite-backed vehicle database with theft tracking
- **Human-in-the-Loop**: Web UI for manual plate corrections
- **Active Learning**: Automatic dataset export for continuous model improvement
- **Docker Deployment**: Multi-container setup for easy cloud deployment
- **REST API**: FastAPI backend with full Swagger documentation
- **React Dashboard**: Modern frontend with real-time results display

## 📊 Architecture

### System Overview

```
Image/Video Input
    ↓
[YOLOv8 Detector] (conf=0.45)
    ↓ detections (bboxes)
[Dynamic Crop & Preprocess]
    ↓ plate images
[EasyOCR Reader] (spatial ordering)
    ↓ raw text
[Fuzzy State Correction]
[Format Scoring]
    ↓ structured results
[SQLite Registry] ← → [React Dashboard]
    ↓
[Human Review/Correction]
    ↓
[Active Learning Export] → [Retrain Pipeline]
```

### Technology Stack

**Backend:**
- **Framework**: FastAPI + Uvicorn
- **ML Engine**: 
  - Detector: YOLOv8s (ultralytics)
  - OCR: EasyOCR with spatial block ordering
  - Heuristics: Fuzzy state matching (difflib), format scoring, dynamic padding
- **Database**: SQLAlchemy + SQLite (data/db/anpr.db)
- **Deployment**: Docker + Docker Compose, NVIDIA GPU support

**Frontend:**
- **Build**: Vite 7.3.1 + React 18.3.1
- **API Client**: TanStack React Query 5.83
- **UI Components**: Shadcn/ui, Radix UI, Framer Motion
- **Forms**: React Hook Form + Zod validation

## 🚀 Quick Start

### Local Development

**Prerequisites:**
- Conda environment with Python 3.11+
- CUDA-capable GPU (optional, CPU fallback supported)
- Node.js 18+

**Backend Setup:**
```bash
conda activate anpr  # or create: conda create -n anpr python=3.11

# Install dependencies
pip install -r requirements.txt

# Start the API server (port 8000)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Access Swagger UI: http://localhost:8000/docs
```

**Frontend Setup:**
```bash
cd frontend

# Install dependencies
npm install

# Start dev server (port 5173)
npm run dev

# Build for production
npm run build
```

### Docker Deployment

**Quick Start:**
```bash
# Build and run all services
docker-compose up --build

# Services will be available at:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

**Production Build:**
```bash
# Create production images without hot-reload
docker build -f Dockerfile.backend -t anpr-backend:prod .
docker build -f Dockerfile.frontend -t anpr-frontend:prod .

# Run with NVIDIA GPU
docker run --gpus all -p 8000:8000 anpr-backend:prod
```

## 📁 Project Structure

```
TheANPR/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # Route handlers + lifecycle
│   │   ├── ml_service.py       # ANPR engine (YOLO + EasyOCR)
│   │   ├── database.py         # SQLAlchemy models
│   │   └── config.py           # Environment configuration
│   ├── pipeline/               # Offline processing
│   │   ├── run_anpr.py         # Image batch processing
│   │   └── run_anpr_video.py   # Video frame processing
│   ├── ocr/
│   │   └── post_process.py     # Heuristic plate correction
│   ├── evaluation/
│   │   └── plate_detector_evaluator.py  # Metrics benchmark
│   ├── data/
│   │   ├── split_plate_dataset.py       # Dataset splitting
│   │   └── rename_hr_plates.py          # Filename normalization
│   └── active_learning/
│       ├── export_corrections.py        # Extract corrections for retraining
│       └── __init__.py
├── frontend/                    # React frontend (Vite)
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── lib/
│   │   │   ├── anpr-api.ts     # Detection API client
│   │   │   └── vehicles-api.ts # Vehicle registry client
│   │   ├── pages/              # Page components
│   │   └── test/               # Frontend tests
│   ├── package.json
│   └── vite.config.ts
├── configs/
│   ├── plate_detector_eval.yaml # Evaluation targets
│   └── plate_detection.yaml    # YOLO dataset config
├── data/
│   ├── db/
│   │   └── anpr.db             # SQLite database
│   ├── raw/                    # Raw images/videos
│   ├── processed/              # Split training data
│   └── active_learning/        # Exported corrections
├── models/
│   └── checkpoints/
│       └── plate_detector_v1.pt # YOLOv8 weights
├── tests/                       # Test suites
│   ├── test_ml_service.py      # Unit tests for ML engine
│   ├── test_main_integration.py # API integration tests
│   └── test_plate_detector_requirements.py
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── Dockerfile.backend          # Backend container image
├── Dockerfile.frontend         # Frontend Nginx container
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔑 Key Algorithms

### Spatial OCR Ordering (Phase 1 Improvement)
Instead of sorting OCR blocks by height alone, we:
1. Group blocks by Y-coordinate proximity (median height ± tolerance)
2. Sort within each row by X-coordinate (left to right)
3. Assemble final plate text respecting spatial layout

**Benefit**: Correctly handles multi-line stacked plates (top line first, then bottom).

### Fuzzy State Correction (Phase 2 Improvement)
When detected state code is invalid:
1. Use `difflib.get_close_matches()` to find closest valid state (cutoff=0.5)
2. Log the auto-correction: "Auto-corrected XX to YY"
3. Apply correction transparently to downstream systems

**Benefit**: Handles OCR confusions (e.g., H→8, R→B) gracefully.

### Dynamic Padding (Phase 3 Improvement)
Crop padding adapts to plate size:
```python
pad_x = max(2, int(box_width * 0.05))  # 5% of width
pad_y = max(2, int(box_height * 0.05)) # 5% of height
```

**Benefit**: Tight crops preserve fine details; loose crops include context.

### Format Scoring (Regression Prevention)
Score plate text based on Indian HSRP rules:
- **1.0**: Perfectly matches `^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$`
- **0.5**: Valid format variant (single-letter state)
- **0.0**: Invalid state prefix or < 8 alphanumeric chars

**Benefit**: Flags suspicious detections for human review.

## 📊 Performance Metrics

**Detection (YOLOv8s):**
- mAP50: 0.9947 (99.47%)
- mAP50-95: 0.8387
- Inference latency: 16.2 ms
- FPS: 61.78

**Database:**
- Schema: DetectionRecord + VehicleRecord
- Indexed on: plate_number, timestamp, status
- Supports: Full-text search, CSV import/export

## 🔄 Correction & Active Learning Workflow

1. **User uploads image** → Detection API returns results with format scores
2. **User reviews & corrects** (if format_score < 0.8)
3. **Correction stored** in DetectionRecord (corrected_plate field)
4. **Run export script**: `python src/active_learning/export_corrections.py`
5. **Exported dataset** goes to `data/active_learning/`
6. **Retrain detector** on corrected samples (coming soon)

## 🧪 Testing

**Run all tests:**
```bash
# Backend unit tests
pytest tests/test_ml_service.py -v

# Backend integration tests
pytest tests/test_main_integration.py -v

# Frontend tests
cd frontend && npm test

# Evaluator regression tests
pytest tests/test_plate_detector_requirements.py -v
```

**CI/CD Pipeline:**
```bash
# GitHub Actions runs on every push:
# - Backend linting (flake8, black)
# - Backend tests (pytest with coverage)
# - Frontend linting (ESLint)
# - Frontend tests (vitest)
# - Docker build validation
# - Security checks (bandit, detect-secrets)
```

## 🛠️ API Endpoints

### Detection
- **POST /api/v1/detect/image** - Upload image for plate detection
  - Returns: `{"success": true, "detections": [...]}`
- **POST /api/v1/detect/correct/{record_id}** - Submit human correction
  - Returns: `{"success": true, "message": "..."}`

### Vehicle Registry
- **GET /api/vehicles** - List all vehicles (filters: status, search, state_code)
- **POST /api/vehicles** - Create new vehicle
- **GET /api/vehicles/record/{plate}** - Get vehicle details
- **PUT /api/vehicles/record/{plate}** - Update vehicle info
- **DELETE /api/vehicles/record/{plate}** - Delete vehicle
- **POST /api/vehicles/record/{plate}/file-complaint** - Mark as stolen
- **POST /api/vehicles/record/{plate}/mark-recovered** - Mark recovered
- **GET /api/vehicles/export** - Export as CSV
- **POST /api/vehicles/import** - Bulk import from CSV

## 📈 Model Versioning (DVC)

To track model changes:
```bash
# Initialize DVC
dvc init

# Add model file
dvc add models/checkpoints/plate_detector_v1.pt

# Commit the .dvc file
git add models/checkpoints/plate_detector_v1.pt.dvc
git commit -m "Track model v1"

# Pull model from remote
dvc pull
```

## 🔐 Security Considerations

**Current Status (Development):**
- ✅ CORS enabled for local frontend
- ⚠️ No authentication on API endpoints
- ⚠️ No rate limiting

**Recommended for Production:**
- [ ] Restrict CORS origins to frontend domain
- [ ] Add JWT authentication for sensitive routes
- [ ] Implement rate limiting (e.g., 1000 req/hour)
- [ ] Validate file uploads (size, MIME type)
- [ ] Add request logging and audit trails

## 🐛 Known Limitations

1. **Video endpoint**: Currently returns stub error response (not implemented)
2. **State code**: Fuzzy matching has 50% similarity threshold (may miss edge cases)
3. **Multi-character OCR**: Assumes max 11 characters (Indian plate standard)
4. **Database**: SQLite scales to ~100k records; PostgreSQL recommended for production

## 📝 Configuration

**Environment Variables** (.env):
```
VITE_API_BASE_URL=http://localhost:8000  # or WSL IP for remote frontend
CUDA_VISIBLE_DEVICES=0                    # GPU device ID
PYTHONUNBUFFERED=1                        # Real-time logging
```

**YAML Configs:**
- `configs/plate_detector_eval.yaml` - Evaluation targets (mAP, latency, VRAM)
- `configs/plate_detection.yaml` - YOLO training dataset metadata


## 👥 Team

Built as a demonstration of modern ML ops practices including:
- Dataset management and splitting
- Model evaluation and benchmarking
- Human-in-the-loop active learning
- Production-grade containerization and CI/CD
- Full-stack ML application development

---

**Last Updated**: 2026-05-11  
**Status**: Production Ready (with noted limitations)  
