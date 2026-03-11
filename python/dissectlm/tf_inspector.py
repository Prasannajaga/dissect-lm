from __future__ import annotations

from typing import Any, Dict


def inspect_tf_runtime() -> Dict[str, Any]:
    try:
        import tensorflow as tf  # type: ignore

        return {
            "available": True,
            "version": tf.__version__,
            "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}
