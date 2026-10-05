"""Shared allow-list contract for generated numeric candidate programs."""

from __future__ import annotations


NUMERIC_OPERATION_RULES_VERSION = "uav-hrl-llm-numeric-operations-v2"

SAFE_BUILTIN_CALLS = frozenset(
    {
        "abs",
        "bool",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "sum",
        "tuple",
        "zip",
    }
)

SAFE_NUMPY_CALLS = frozenset(
    {
        "np.abs",
        "np.all",
        "np.any",
        "np.arange",
        "np.array",
        "np.asarray",
        "np.bool_",
        "np.clip",
        "np.concatenate",
        "np.copy",
        "np.count_nonzero",
        "np.exp",
        "np.float32",
        "np.float64",
        "np.int32",
        "np.int64",
        "np.isfinite",
        "np.log",
        "np.log1p",
        "np.max",
        "np.maximum",
        "np.mean",
        "np.median",
        "np.min",
        "np.minimum",
        "np.ones",
        "np.ravel",
        "np.reshape",
        "np.sqrt",
        "np.stack",
        "np.std",
        "np.sum",
        "np.transpose",
        "np.var",
        "np.where",
        "np.zeros",
        "np.linalg.norm",
    }
)

SAFE_NUMPY_ATTRIBUTES = frozenset(
    {"np.float32", "np.float64", "np.int32", "np.int64", "np.bool_"}
)

SAFE_ARRAY_METHODS = frozenset(
    {
        "astype",
        "sum",
        "mean",
        "min",
        "max",
        "std",
        "var",
        "all",
        "any",
        "clip",
        "reshape",
        "flatten",
        "ravel",
        "transpose",
        "copy",
    }
)

SAFE_ARRAY_READ_ATTRIBUTES = frozenset({"shape", "size", "ndim"})
SAFE_LOCAL_CONTAINER_METHODS = frozenset({"append"})

ARRAY_VIEW_METHODS = frozenset({"reshape", "ravel", "transpose"})
ARRAY_COPY_METHODS = frozenset({"copy", "flatten", "clip"})
ARRAY_REDUCTION_METHODS = frozenset(
    {"sum", "mean", "min", "max", "std", "var", "all", "any"}
)

NUMPY_VIEW_CALLS = frozenset({"np.reshape", "np.ravel", "np.transpose"})
NUMPY_COPY_CALLS = frozenset({"np.copy"})

ALLOWED_DTYPE_STRINGS = frozenset(
    {
        "bool",
        "float",
        "float32",
        "float64",
        "int",
        "int32",
        "int64",
    }
)
ALLOWED_DTYPE_NAMES = frozenset({"bool", "float", "int"})
UNSAFE_ARRAY_KEYWORDS = frozenset({"out", "like"})
