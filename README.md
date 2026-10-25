# ThermoSR — Thermal Image Super-Resolution Lab

> **Academic Deep Learning Platform** for guided thermal infrared super-resolution with **Reliability-Aware RGB Guidance**.

---

## 1. Project Architecture

```
ThermoSR/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI Application Entrypoint
│   │   ├── api/routes/                 # REST Endpoints (health, system, datasets, reconstruction, training, experiments, models)
│   │   ├── core/                       # Configuration, Dynamic Paths & Structured Logging
│   │   ├── db/                         # SQLAlchemy 2 Models & Session
│   │   ├── schemas/                    # Pydantic v2 Validation Schemas
│   │   ├── services/                   # Preprocessing, Metrics, Datasets, Artifacts
│   │   ├── ml/
│   │   │   ├── models/                 # Model A (Baseline), Model B (RGB Fusion), Model C (Reliability Gate)
│   │   │   ├── losses/                 # Composite Thermal Loss (L1, Sobel Edge, Downsampling Cycle)
│   │   │   ├── training/               # TrainingSession, Cosine Scheduler, Checkpoints
│   │   │   └── inference/              # Inference Pipeline & Heatmap Generation
│   │   └── workers/                    # Background Jobs & Cancellation Registry
│   ├── alembic/                        # Database Migrations
│   ├── tests/                          # Pytest Automated Test Suite (11/11 passing)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── layouts/                    # Clean Light-Theme Layout & Sidebar
│   │   ├── pages/                      # Dashboard, Reconstruct, Dataset, TrainingLab, Experiments, Models, Settings
│   │   ├── services/                   # Axios API Client
│   │   └── types/                      # TypeScript Strict Models
│   ├── package.json
│   └── vite.config.ts
├── data/                               # Sample Paired Test Data & Local Datasets
├── artifacts/                          # Checkpoints, Uploads & Visualizations
├── docs/                               # Research Methodology & Paper Documentation
└── docker-compose.yml
```

---

## 2. Running the Application (Windows PowerShell)

### Terminal 1 — Start the Backend (Port 8000)

```powershell
Set-Location "D:\software Engineering (academics)\Projects\ThermoSR\backend"
python -m alembic upgrade head
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API & Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

### Terminal 2 — Start the Frontend (Port 3000)

```powershell
Set-Location "D:\software Engineering (academics)\Projects\ThermoSR\frontend"
npm run dev
```
- Frontend Web UI: `http://localhost:3000`

---

## 3. Running Automated Tests

```powershell
Set-Location "D:\software Engineering (academics)\Projects\ThermoSR\backend"
python -m pytest tests/test_backend.py -v
```

---

## 4. UI Features & Controls

- **Clean Light Theme**: Intuitive academic design built with slate and crisp typography.
- **Before / After Comparison Buttons**: Instead of cumbersome sliders, view results cleanly with dedicated `[Before (LR Input)]`, `[After (Reconstructed)]`, and `[Side-by-Side]` buttons, plus a quick `[Flip ⇄]` button for instantaneous toggle.
- **Four Visualization Modes**: Switch seamlessly between **False Color** (Inferno), **Native Grayscale**, **Learned Reliability Heatmap**, and **Error Heatmap**.
- **Dataset Inspector**: Scan, pair, and inspect all 1,001 FLIR challenge images at `c:\Users\abdur\Downloads\challengedataset - Copy\challengedataset`.
