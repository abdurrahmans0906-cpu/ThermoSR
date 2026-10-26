# ThermoSR — Thermal Image Super-Resolution Lab

> **Academic Deep Learning Platform** for guided thermal infrared super-resolution with **Reliability-Aware RGB Guidance** and an **Intelligent QA & Experiment Advisor (Laya System 1)**.

---

## 1. Project Architecture

```
ThermoSR/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI Application Entrypoint
│   │   ├── api/routes/                 # REST Endpoints (health, system, datasets, reconstruction, training, experiments, models, qa)
│   │   ├── core/                       # Configuration, Dynamic Paths & Structured Logging
│   │   ├── db/                         # SQLAlchemy 2 Models & Session
│   │   ├── schemas/                    # Pydantic v2 Validation Schemas
│   │   ├── services/                   # Preprocessing, Metrics, Datasets, Artifacts
│   │   ├── ml/
│   │   │   ├── models/                 # Model A (Baseline), Model B (RGB Fusion), Model C (Reliability Gate)
│   │   │   ├── losses/                 # Composite Thermal Loss (L1, Sobel Edge, Downsampling Cycle)
│   │   │   ├── training/               # TrainingSession, Cosine Scheduler, Checkpoints
│   │   │   └── inference/              # Inference Pipeline & Heatmap Generation
│   │   ├── qa/                         # Laya System 1 QA & Experiment Advisor
│   │   │   ├── schemas.py              # Strict Pydantic QA Schemas
│   │   │   ├── metadata_checks.py      # Deterministic Validation & Mathematical Constraints
│   │   │   ├── laya_adapter.py         # Laya System 1 Non-Autoregressive Decision Adapter
│   │   │   └── experiment_advisor.py   # Advisory Experiment Recommendations
│   │   └── workers/                    # Background Jobs & Cancellation Registry
│   ├── alembic/                        # Database Migrations
│   ├── tests/                          # 34 Automated Tests (Models, Losses, Metrics, QA Schemas, Advisor, API)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/                 # LayaQAPanel & Custom Widgets
│   │   ├── layouts/                    # Clean Light-Theme Layout & Sidebar
│   │   ├── pages/                      # Dashboard, Reconstruct, Dataset, TrainingLab, Experiments, Models, Settings
│   │   ├── services/                   # Axios API Client & Laya QA Service
│   │   └── types/                      # TypeScript Strict Models & QA Types
│   ├── package.json
│   └── vite.config.js
├── data/                               # Sample Paired Test Data & Local Datasets
├── docs/                               # Research Methodology & Paper Documentation
├── run_thermosr.bat                    # One-Click Windows Launcher (CMD)
├── run_thermosr.ps1                    # One-Click Windows Launcher (PowerShell)
└── docker-compose.yml
```

---

## 2. Quick Start: Running the Application

### Option A: One-Click Launcher (Recommended)
Simply double-click **`run_thermosr.bat`** (or in PowerShell, run `.\run_thermosr.ps1`).  
This automatically starts both the FastAPI backend and React frontend in separate windows.

### Option B: Manual PowerShell Terminals

#### Terminal 1 — Start the Backend (Port 8000)
```powershell
Set-Location "D:\software Engineering (academics)\Projects\ThermoSR\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API & Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

#### Terminal 2 — Start the Frontend (Port 3000)
```powershell
Set-Location "D:\software Engineering (academics)\Projects\ThermoSR\frontend"
npm run dev
```
- Frontend Web UI: `http://localhost:3000`

---

## 3. Running Automated Tests

Run the complete test suite (original deep learning models + Laya QA add-on):
```powershell
Set-Location "D:\software Engineering (academics)\Projects\ThermoSR\backend"
python -m pytest tests/ -v
```
*(All 34 unit and integration tests execute and pass in under 15 seconds).*

---

## 4. UI Features & Controls

- **Clean Light Theme**: Intuitive academic design built with slate and crisp typography.
- **Before / After Comparison Buttons**: Dedicated `[Before (LR Input)]`, `[After (Reconstructed)]`, and `[Side-by-Side]` buttons, plus a quick `[Flip ⇄]` button for instantaneous comparison.
- **Four Visualization Modes**: Switch between **False Color** (Inferno), **Native Grayscale**, **Learned Reliability Heatmap**, and **Error Heatmap**.
- **Dataset Inspector**: Scan, pair, and inspect challenge images at `c:\Users\abdur\Downloads\challengedataset - Copy\challengedataset`.

---

## 5. Intelligent QA & Experiment Advisor (Laya System 1)

The application includes an optional AI-assisted quality assessment add-on powered by **Laya System 1** (a non-autoregressive decision engine):

- **Metadata Quality Assurance (`POST /api/qa/laya-assess`)**:
  - Deterministically verifies image dimensions, scale factors ($\times 2, \times 4, \times 8, \times 16$), sensor optical registration status, and metric availability.
  - Generates an evaluation review status: `ready_for_review`, `human_review`, `insufficient_evidence`, or `unavailable`.
- **Experiment Recommendations (`POST /api/qa/experiment-recommendation`)**:
  - Recommends which ablation study or baseline run to inspect next (`thermal_baseline`, `rgb_fusion`, `reliability_fusion`, `rerun_evaluation`, `human_review`).
- **Safety Invariants**:
  - Purely read-only: guaranteed `reconstruction_modified: false` and `executed: false`.
  - Deterministic mathematical constraints strictly override decision model outputs.
  - Graceful fallback: functions completely via deterministic rules if Laya is uninstalled or offline.
