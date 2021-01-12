from multiprocessing import cpu_count as _cpu_count

from ._to_tensor import to_tensor
from .data import *


def _get_num_workers(num_workers):
    if num_workers == -1:
        return _cpu_count() - 1
    return num_workers