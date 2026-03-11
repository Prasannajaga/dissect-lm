from __future__ import annotations

from typing import Any, Dict


def inspect_torch_runtime() -> Dict[str, Any]:
    try:
        import torch  # type: ignore

        return {
            "available": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}
