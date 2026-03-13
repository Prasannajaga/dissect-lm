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


def inspect_checkpoint(checkpoint: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mode": "deep",
        "checkpoint": checkpoint,
        "frameworks": {},
    }

    try:
        import torch  # type: ignore

        out["frameworks"]["torch"] = torch.__version__
        data = torch.load(checkpoint, map_location="cpu")

        payload = data
        if isinstance(data, dict):
            if "state_dict" in data and isinstance(data["state_dict"], dict):
                payload = data["state_dict"]
            elif "model" in data and isinstance(data["model"], dict):
                payload = data["model"]

        if isinstance(payload, dict):
            tensor_items = []
            total_params = 0
            for key, value in payload.items():
                shape = tuple(getattr(value, "shape", ()))
                if not shape:
                    continue
                params = 1
                for dim in shape:
                    params *= int(dim)
                total_params += params
                tensor_items.append((str(key), list(shape), params))

            tensor_items.sort(key=lambda x: x[2], reverse=True)
            out["checkpoint_summary"] = {
                "tensor_count": len(tensor_items),
                "total_params": total_params,
                "top_tensors": [
                    {"name": name, "shape": shape, "params": params}
                    for name, shape, params in tensor_items[:20]
                ],
            }
        else:
            out["checkpoint_summary"] = {
                "note": f"Unsupported checkpoint payload type: {type(payload).__name__}"
            }
    except Exception as exc:  # pragma: no cover
        out["frameworks"]["torch"] = f"error: {exc}"

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep model inspector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model")
    group.add_argument("--checkpoint")
    args = parser.parse_args()

    if args.checkpoint:
        print(json.dumps(inspect_checkpoint(args.checkpoint)))
    else:
        print(json.dumps(inspect_model(args.model)))


if __name__ == "__main__":
    main()
