import asyncio
import threading
from typing import Dict, Any, Optional
from app.core.logging import logger

# Global registries for running jobs
active_training_sessions: Dict[str, Any] = {}
active_reconstruction_jobs: Dict[str, Any] = {}

def register_training_session(run_id: str, session: Any):
    active_training_sessions[run_id] = session

def unregister_training_session(run_id: str):
    if run_id in active_training_sessions:
        del active_training_sessions[run_id]

def cancel_training_session(run_id: str) -> bool:
    if run_id in active_training_sessions:
        session = active_training_sessions[run_id]
        if hasattr(session, "cancel"):
            session.cancel()
            return True
    return False

def register_reconstruction_job(run_id: str, job_meta: Any):
    active_reconstruction_jobs[run_id] = job_meta

def unregister_reconstruction_job(run_id: str):
    if run_id in active_reconstruction_jobs:
        del active_reconstruction_jobs[run_id]

def cancel_reconstruction_job(run_id: str) -> bool:
    if run_id in active_reconstruction_jobs:
        job = active_reconstruction_jobs[run_id]
        if isinstance(job, dict):
            job["cancelled"] = True
            return True
    return False
