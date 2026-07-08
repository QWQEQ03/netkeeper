from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when torch is installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - current test environment.
    torch = None


Tensor = Any


def tensor(data: Any, dtype: str = "float32") -> Tensor:
    if torch is not None:
        torch_dtype = {
            "bool": torch.bool,
            "float32": torch.float32,
            "int64": torch.int64,
        }[dtype]
        return torch.tensor(data, dtype=torch_dtype)

    np_dtype = {
        "bool": np.bool_,
        "float32": np.float32,
        "int64": np.int64,
    }[dtype]
    return np.asarray(data, dtype=np_dtype)


def zeros(shape: tuple[int, ...], dtype: str = "float32") -> Tensor:
    if torch is not None:
        torch_dtype = {
            "bool": torch.bool,
            "float32": torch.float32,
            "int64": torch.int64,
        }[dtype]
        return torch.zeros(shape, dtype=torch_dtype)

    np_dtype = {
        "bool": np.bool_,
        "float32": np.float32,
        "int64": np.int64,
    }[dtype]
    return np.zeros(shape, dtype=np_dtype)


def ones(shape: tuple[int, ...], dtype: str = "float32") -> Tensor:
    if torch is not None:
        torch_dtype = {
            "bool": torch.bool,
            "float32": torch.float32,
            "int64": torch.int64,
        }[dtype]
        return torch.ones(shape, dtype=torch_dtype)

    np_dtype = {
        "bool": np.bool_,
        "float32": np.float32,
        "int64": np.int64,
    }[dtype]
    return np.ones(shape, dtype=np_dtype)


def to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def is_tensor(value: Any) -> bool:
    if torch is not None and isinstance(value, torch.Tensor):
        return True
    return isinstance(value, np.ndarray)
