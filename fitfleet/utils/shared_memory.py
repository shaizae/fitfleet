"""Numpy arrays in named shared memory blocks, shared between the parent and pool workers.

`_ATTACHED` holds the blocks this worker process has already attached to, keyed by name.
"""
from __future__ import annotations

import multiprocessing
import shutil
from functools import wraps
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Any, ParamSpec, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

P = ParamSpec("P")
R = TypeVar("R")


class SharedMemory:
    """A numpy array stored in a named shared memory block, owned by this process.

    Every instance registers itself in a class-level registry so `cleanup()` can
    release all blocks at once. Pickling sends only (name, shape, dtype); workers
    reattach by name, once per process.
    """
    _shms: dict[str, SharedMemory] = {}

    def __init__(self, memory: shared_memory.SharedMemory, shape: Tuple[Any, ...], dtype: np.dtype, ) -> None:
        """Wrap an already created block and register it; raises if the name is taken."""
        self.shm = memory
        self.shape = shape
        self.dtype = dtype

        SharedMemory.add(self)

    def unlink(self) -> None:
        """Close and destroy the block and drop it from the registry."""
        self.shm.close()
        self.shm.unlink()

        SharedMemory._shms.pop(self.shm.name, None)

    @staticmethod
    def add(shm: SharedMemory) -> None:
        """Register a block by name; raises ValueError if the name is already registered."""
        if shm.shm.name in SharedMemory._shms.keys():
            raise ValueError(f"{shm.shm.name} is already exists")
        SharedMemory._shms[shm.shm.name] = shm

    @staticmethod
    def validate(name: str) -> bool:
        """Whether a block with this name is registered in this process."""
        return name in list(SharedMemory._shms.keys())

    @staticmethod
    def cleanup() -> None:
        """Unlink every registered block."""
        for shm in list(SharedMemory._shms.values()):
            shm.unlink()

        SharedMemory._shms.clear()

    @property
    def array(self) -> np.ndarray:
        """A numpy view on the block (no copy)."""
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf, )

    def __repr__(self) -> str:
        """Name, shape and dtype of the block."""
        return (f"SharedMemory(name={self.shm.name}, shape={self.shape}, dtype={self.dtype})")

    def __deepcopy__(self, memo: dict[int, Any]) -> SharedMemory:
        """Return self: the block is shared by design, so copying a task must not reopen it."""
        return self

    def __reduce__(self) -> tuple[Callable[[str, Tuple[Any, ...], str], SharedMemory], tuple[str, Tuple[Any, ...], str]]:
        """Pickle only (name, shape, dtype); workers reattach by name, once per process."""
        return _attach_shared, (self.shm.name, self.shape, np.dtype(self.dtype).str)


_ATTACHED: dict[str, SharedMemory] = {}


def _attach_shared(name: str, shape: Tuple[Any, ...], dtype: str) -> SharedMemory:
    """Unpickle a SharedMemory: reattach to the named block without taking ownership.

    The owning process gets its own registered instance back. Handles are cached only
    inside pool workers: the parent reuses names ("features", "target") across runs, so
    a cached handle there would point at an unlinked block. `dtype` is a numpy dtype
    string. The attached instance skips __init__ because workers don't own the block,
    and `track=False` (Python 3.13+) keeps the resource tracker from unlinking it.
    """
    owned = SharedMemory._shms.get(name)
    if owned is not None:
        return owned

    in_worker = multiprocessing.parent_process() is not None
    cached = _ATTACHED.get(name) if in_worker else None
    if cached is not None and cached.shape == tuple(shape) and cached.dtype == np.dtype(dtype):
        return cached
    if in_worker:
        _release_other_runs(name)

    try:
        memory = shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        memory = shared_memory.SharedMemory(name=name)
    attached = SharedMemory.__new__(SharedMemory)
    attached.shm = memory
    attached.shape = tuple(shape)
    attached.dtype = np.dtype(dtype)
    if in_worker:
        _ATTACHED[name] = attached
    return attached


def _run_of(name: str) -> str | None:
    """The run id in a name made by run_block_name() ("ff_<run id>_<kind>"), else None."""
    parts = name.split("_")
    return parts[1] if len(parts) >= 3 and parts[0] == "ff" else None


def _release_other_runs(name: str) -> None:
    """A reused worker drops its handles to blocks from earlier runs. On Windows a
    block stays allocated while any process holds a handle, even after unlink.
    A handle that still has a live numpy view can't be closed now; it closes with the view."""
    run = _run_of(name)
    if run is None:
        return
    for old in [n for n in _ATTACHED if _run_of(n) not in (None, run)]:
        block = _ATTACHED.pop(old)
        try:
            block.shm.close()
        except BufferError:
            pass


def run_block_name(run_id: str, kind: str) -> str:
    """Unique name per run, so concurrent pipelines (or two runs on one machine)
    never collide on a shared memory block."""
    return f"ff_{run_id}_{kind}"  # short: macOS caps names at 31 chars


SHM_DIR = Path("/dev/shm")


def check_shared_memory_space(nbytes: int) -> None:
    """On Linux, shared memory lives in /dev/shm. Writing past its size doesn't raise: the
    process is killed with SIGBUS. Docker limits /dev/shm to 64 MB unless --shm-size is set.
    Windows and macOS have no fixed-size shared memory filesystem, so nothing is checked there."""
    if not SHM_DIR.is_dir():
        return
    free = shutil.disk_usage(SHM_DIR).free
    if nbytes > free:
        raise MemoryError(
            f"Not enough shared memory: the data needs {nbytes / 2 ** 20:,.0f} MB but {SHM_DIR} has "
            f"{free / 2 ** 20:,.0f} MB free. In Docker, raise the container's shm_size (--shm-size).")


def create_shared_memory(arr: np.ndarray, name: str) -> SharedMemory:
    """Copy `arr` into a new shared memory block called `name` and return its owner handle.

    Raises ValueError if the name is already in use and MemoryError if /dev/shm is too small.
    """
    if SharedMemory.validate(name):
        raise ValueError(f"{name} is already exists")
    check_shared_memory_space(arr.nbytes)
    memory = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    shared_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=memory.buf)
    shared_array[:] = arr[:]
    shared = SharedMemory(memory=memory, shape=arr.shape, dtype=arr.dtype)
    return shared


def features_unpackage(shared_memory: SharedMemory, index: Sequence[int] | np.ndarray,
                       features_index: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
    """Rows `index`, columns `features_index` (None = all) in a single copy."""
    array = shared_memory.array
    if features_index is None:
        return array[index]
    return array[np.ix_(np.asarray(index), np.asarray(features_index))]


def cleanup_shared_memory(func: Callable[P, R]) -> Callable[P, R]:
    """Unlink every registered shared memory block once `func` returns or raises."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Call `func`, then release all shared memory blocks."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            SharedMemory.cleanup()

    return wrapper
