"""
GPU-Erkennung für den BW Colorizer.
Erkennt NVIDIA-GPUs über torch/CUDA und liefert einen sprechenden Namen
sowie den zu verwendenden torch-device-String.
"""

from __future__ import annotations
import subprocess


def detect_gpu() -> dict:
    """
    Gibt ein Dict zurück:
      {
        "available": bool,
        "name": str,           # z.B. "NVIDIA GeForce RTX 5080"
        "device": str,         # "cuda" oder "cpu"
        "vram_gb": float | None
      }
    """
    result = {
        "available": False,
        "name": "Keine GPU erkannt (CPU-Modus)",
        "device": "cpu",
        "vram_gb": None,
    }

    try:
        import torch  # noqa: WPS433 (lazy import, torch kann groß/optional sein)
    except ImportError:
        result["name"] = "PyTorch nicht installiert"
        return result

    if torch.cuda.is_available():
        try:
            idx = 0
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            vram_gb = round(props.total_memory / (1024 ** 3), 1)
            result.update(
                available=True,
                name=name,
                device="cuda",
                vram_gb=vram_gb,
            )
        except Exception:
            result.update(available=True, name="NVIDIA GPU (Details nicht lesbar)", device="cuda")
    else:
        # Fallback: nvidia-smi direkt abfragen, falls torch ohne CUDA-Build installiert wurde
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            if out:
                first_line = out.splitlines()[0]
                result["name"] = f"{first_line} (torch ohne CUDA-Support installiert!)"
        except Exception:
            pass

    return result


if __name__ == "__main__":
    info = detect_gpu()
    print(f"GPU verfügbar: {info['available']}")
    print(f"Name: {info['name']}")
    print(f"Device: {info['device']}")
    print(f"VRAM: {info['vram_gb']} GB")
