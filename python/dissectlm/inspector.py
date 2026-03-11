from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def _safe_get(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def inspect_model(model: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mode": "deep",
        "model": model,
        "frameworks": {},
    }

    try:
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        out["config"] = {
            "class": cfg.__class__.__name__,
            "model_type": _safe_get(cfg, "model_type"),
            "hidden_size": _safe_get(cfg, "hidden_size") or _safe_get(cfg, "n_embd"),
            "num_layers": _safe_get(cfg, "num_hidden_layers") or _safe_get(cfg, "n_layer"),
            "num_heads": _safe_get(cfg, "num_attention_heads") or _safe_get(cfg, "n_head"),
            "num_key_value_heads": _safe_get(cfg, "num_key_value_heads")
            or _safe_get(cfg, "num_kv_heads")
            or _safe_get(cfg, "n_head_kv"),
        }
        out["frameworks"]["transformers"] = "available"
    except Exception as exc:  # pragma: no cover
        out["frameworks"]["transformers"] = f"error: {exc}"

    try:
        import torch  # type: ignore

        out["frameworks"]["torch"] = torch.__version__
    except Exception as exc:  # pragma: no cover
        out["frameworks"]["torch"] = f"error: {exc}"

    try:
        import tensorflow as tf  # type: ignore

        out["frameworks"]["tensorflow"] = tf.__version__
    except Exception as exc:  # pragma: no cover
        out["frameworks"]["tensorflow"] = f"error: {exc}"

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep model inspector")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print(json.dumps(inspect_model(args.model)))


if __name__ == "__main__":
    main()
