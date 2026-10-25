import os
import torch
from fastapi import APIRouter
from app.schemas.system import DeviceInfo

router = APIRouter(prefix="/system", tags=["System"])

@router.get("/device", response_model=DeviceInfo)
def get_device_info():
    cuda_avail = torch.cuda.is_available()
    device_str = "cuda" if cuda_avail else "cpu"
    device_name = "CPU"
    cuda_ver = None
    vram_tot = None
    vram_used = None
    vram_free = None
    recommended_batch = 8
    notes = []

    if cuda_avail:
        try:
            device_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            vram_tot = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
            vram_used = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
            vram_free = round(vram_tot - vram_used, 1)

            if vram_tot < 4500:
                recommended_batch = 8
                notes.append("GPU VRAM <= 4GB detected. Recommended patch size: 64x64, batch size: 4-8.")
            elif vram_tot < 8500:
                recommended_batch = 16
            else:
                recommended_batch = 32
        except Exception as e:
            notes.append(f"CUDA query error: {str(e)}")
    else:
        notes.append("PyTorch running on CPU. Training will use multi-threaded CPU execution. Batch size 4-8 recommended.")
        recommended_batch = 4

    return DeviceInfo(
        device=device_str,
        device_name=device_name,
        cuda_available=cuda_avail,
        cuda_version=cuda_ver,
        vram_total_mb=vram_tot,
        vram_used_mb=vram_used,
        vram_free_mb=vram_free,
        torch_version=torch.__version__,
        cpu_cores=os.cpu_count() or 4,
        recommended_batch_size=recommended_batch,
        notes=notes
    )
