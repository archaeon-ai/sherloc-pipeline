"""Installed implementation and numerical runtime identity, without Git access."""

import hashlib
import json
import platform
from importlib.resources import files

import numpy
import scipy

IMPLEMENTATION_FILES = (
    "core/identification/__init__.py",
    "core/identification/compatibility.py",
    "core/identification/decision.py",
    "core/identification/inspection.py",
    "core/identification/mixture.py",
    "core/identification/pearson.py",
    "core/identification/prep.py",
    "core/identification/provenance.py",
    "core/identification/recipe.py",
    "database/reference_models.py",
    "models/feature_policy.py",
    "models/identification.py",
    "models/references.py",
    "runtime/worker.py",
    "services/identification.py",
    "services/identification_runtime.py",
    "services/identification_source.py",
    "services/reference_import.py",
    "services/reference_serving.py",
    "services/reference_store.py",
)


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def implementation_fingerprint() -> str:
    root = files("sherloc_pipeline")
    return fingerprint({name: hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest()
                        for name in IMPLEMENTATION_FILES})


def runtime_provenance() -> dict:
    environment = {"python": platform.python_version(), "numpy": numpy.__version__,
                   "scipy": scipy.__version__}
    return {"code_sha256": implementation_fingerprint(),
            "code_identity_basis": "installed_source_bytes", "environment": environment,
            "environment_sha256": fingerprint(environment)}
