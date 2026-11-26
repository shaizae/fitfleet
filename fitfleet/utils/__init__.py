"""Shared utilities, grouped by topic. The public names are re-exported here, so
`from fitfleet.utils import X` keeps working.

- shared_memory: numpy arrays in shared memory blocks for the worker pool
- enums: ValidationType, the filtering criteria, ReportFormat and `in_enum`
- sound: the end-of-run sounds and the WAV player
- decorators: `force_gc`
- models: `is_fitted` and `create_path`
- postgres: reading features and target from PostgreSQL
- feature_selection: helpers behind BaseML's feature selection
"""
from fitfleet.utils.decorators import GC_EVERY_N_TASKS, force_gc
from fitfleet.utils.enums import (FilteringCriteriaClassification, FilteringCriteriaRegression, ReportFormat,
                                  ValidationType, in_enum)
from fitfleet.utils.models import create_path, is_fitted
from fitfleet.utils.shared_memory import (SharedMemory, check_shared_memory_space, cleanup_shared_memory,
                                          create_shared_memory, features_unpackage, run_block_name)
from fitfleet.utils.sound import Sound

__all__ = [
    "GC_EVERY_N_TASKS", "force_gc",
    "FilteringCriteriaClassification", "FilteringCriteriaRegression", "ReportFormat", "ValidationType", "in_enum",
    "create_path", "is_fitted",
    "SharedMemory", "check_shared_memory_space", "cleanup_shared_memory", "create_shared_memory",
    "features_unpackage", "run_block_name",
    "Sound",
]
