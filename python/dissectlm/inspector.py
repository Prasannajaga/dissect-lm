from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

TOP_TENSOR_LIMIT = 20
MAX_SCAN_NODES = 200_000


def _safe_get(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def _load_torch_module() -> Tuple[Optional[Any], str]:
    try:
        import torch  # type: ignore

        return torch, torch.__version__
    except Exception as exc:  # pragma: no cover
        return None, f"error: {exc}"


def _load_tensorflow_module() -> Tuple[Optional[Any], str]:
    try:
        import tensorflow as tf  # type: ignore

        return tf, tf.__version__
    except Exception as exc:  # pragma: no cover
        return None, f"error: {exc}"


def _shape_from_tensor_like(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dims = [int(dim) for dim in tuple(shape)]
    except Exception:
        return None
    if not dims:
        return None
    return dims


def _prod(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _collect_tensor_items(root: Any) -> Tuple[List[Tuple[str, List[int], int]], bool]:
    tensor_items: List[Tuple[str, List[int], int]] = []
    stack: List[Tuple[str, Any]] = [("", root)]
    seen: set[int] = set()
    visited = 0
    truncated = False

    while stack:
        path, node = stack.pop()
        visited += 1
        if visited > MAX_SCAN_NODES:
            truncated = True
            break

        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)

        shape = _shape_from_tensor_like(node)
        if shape is not None:
            name = path or "<root>"
            tensor_items.append((name, shape, _prod(shape)))
            continue

        if isinstance(node, Mapping):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                stack.append((child, value))
            continue

        if isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            for idx, value in enumerate(node):
                child = f"{path}[{idx}]" if path else f"[{idx}]"
                stack.append((child, value))
            continue

        state_dict_fn = getattr(node, "state_dict", None)
        if callable(state_dict_fn):
            try:
                state_dict = state_dict_fn()
            except Exception:
                state_dict = None
            if isinstance(state_dict, Mapping):
                for key, value in state_dict.items():
                    base = f"{path}.state_dict" if path else "state_dict"
                    stack.append((f"{base}.{key}", value))

        attrs = getattr(node, "__dict__", None)
        if isinstance(attrs, dict):
            for key, value in attrs.items():
                if key.startswith("__"):
                    continue
                child = f"{path}.{key}" if path else str(key)
                stack.append((child, value))

    return tensor_items, truncated


def _summarize_tensor_items(
    tensor_items: List[Tuple[str, List[int], int]], framework: str, truncated: bool = False
) -> Dict[str, Any]:
    tensor_items.sort(key=lambda x: x[2], reverse=True)
    summary: Dict[str, Any] = {
        "framework": framework,
        "tensor_count": len(tensor_items),
        "total_params": sum(item[2] for item in tensor_items),
        "top_tensors": [
            {"name": name, "shape": shape, "params": params}
            for name, shape, params in tensor_items[:TOP_TENSOR_LIMIT]
        ],
    }
    if truncated:
        summary["note"] = (
            f"Traversal stopped early after scanning {MAX_SCAN_NODES} nodes; "
            "increase MAX_SCAN_NODES if needed."
        )
    return summary


def _safe_torch_load(torch: Any, checkpoint: str) -> Any:
    try:
        # torch>=2.6 supports weights_only; explicit False keeps broader checkpoint support.
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint, map_location="cpu")


def _inspect_torch_checkpoint(torch: Any, checkpoint: str) -> Dict[str, Any]:
    data = _safe_torch_load(torch, checkpoint)
    tensor_items, truncated = _collect_tensor_items(data)
    if not tensor_items:
        return {
            "framework": "torch",
            "note": f"No tensor-like payload found in checkpoint root type: {type(data).__name__}",
            "tensor_count": 0,
            "total_params": 0,
            "top_tensors": [],
        }
    return _summarize_tensor_items(tensor_items, framework="torch", truncated=truncated)


def _normalize_tf_checkpoint_path(checkpoint: str) -> str:
    if checkpoint.endswith(".index"):
        return checkpoint[: -len(".index")]
    return checkpoint


def _inspect_tensorflow_checkpoint(tf: Any, checkpoint: str) -> Dict[str, Any]:
    ckpt_path = _normalize_tf_checkpoint_path(checkpoint)
    variables = tf.train.list_variables(ckpt_path)
    tensor_items: List[Tuple[str, List[int], int]] = []
    for name, shape in variables:
        dims = [int(dim) for dim in shape]
        if not dims:
            continue
        tensor_items.append((str(name), dims, _prod(dims)))

    if not tensor_items:
        return {
            "framework": "tensorflow",
            "note": "No tensor variables discovered in TensorFlow checkpoint.",
            "tensor_count": 0,
            "total_params": 0,
            "top_tensors": [],
        }
    return _summarize_tensor_items(tensor_items, framework="tensorflow")


def _preferred_framework_order(checkpoint: str) -> List[str]:
    name = os.path.basename(checkpoint).lower()
    ext = os.path.splitext(name)[1]
    if name.endswith(".index") or ".data-" in name or ext in {".h5", ".hdf5"}:
        return ["tensorflow", "torch"]
    if ext in {".pt", ".pth", ".bin"}:
        return ["torch", "tensorflow"]
    return ["torch", "tensorflow"]


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

    _, torch_status = _load_torch_module()
    _, tf_status = _load_tensorflow_module()
    out["frameworks"]["torch"] = torch_status
    out["frameworks"]["tensorflow"] = tf_status

    return out


def inspect_checkpoint(checkpoint: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mode": "deep",
        "checkpoint": checkpoint,
        "frameworks": {},
    }

    torch_mod, torch_status = _load_torch_module()
    tf_mod, tf_status = _load_tensorflow_module()
    out["frameworks"]["torch"] = torch_status
    out["frameworks"]["tensorflow"] = tf_status

    attempted: Dict[str, str] = {}
    for framework in _preferred_framework_order(checkpoint):
        if framework == "torch" and torch_mod is not None:
            try:
                out["checkpoint_summary"] = _inspect_torch_checkpoint(
                    torch_mod, checkpoint
                )
                return out
            except Exception as exc:  # pragma: no cover
                attempted["torch"] = str(exc)
        elif framework == "tensorflow" and tf_mod is not None:
            try:
                out["checkpoint_summary"] = _inspect_tensorflow_checkpoint(
                    tf_mod, checkpoint
                )
                return out
            except Exception as exc:  # pragma: no cover
                attempted["tensorflow"] = str(exc)

    out["checkpoint_summary"] = {
        "note": (
            "Could not parse checkpoint with available frameworks. "
            "Install/verify torch or tensorflow and ensure the checkpoint format matches."
        ),
        "attempted": attempted,
    }

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
