import torch
import numpy as np
import time
from multiprocessing import cpu_count

import os, sys

def _to_tensor_leaf(arg):
    if isinstance(arg, np.ndarray):
        arg = torch.from_numpy(arg)
    elif not isinstance(arg, torch.Tensor):
        raise TypeError(f'Only np.ndarray and torch.Tensor are supported. Got {type(arg)}.')

    if arg.dtype == torch.float64:
        print('[Warning] Auto convert float64 to float32, this could lead to extra memory usage.')
        arg = arg.float()
    elif arg.dtype == torch.int32:
        print('[Warning] Auto convert int32 to int64, this could lead to extra memory usage.')
        arg = arg.long()
    return arg

def _deep_to_tensor(arg):
    if isinstance(arg, list) or isinstance(arg, tuple):
        return type(arg)([_deep_to_tensor(a) for a in arg])
    else:
        return _to_tensor_leaf(arg)

def to_tensor(*args):
    rt = [_deep_to_tensor(arg) for arg in args]            
    return rt[0] if len(rt) == 1 else rt


def _get_num_workers(num_workers):
    if num_workers == -1:
        return cpu_count() - 1
    return num_workers



class Progbar(object):
    def __init__(self, target, width=30, interval=0.05):
        self.target = target
        self.width = width
        self.interval = interval

        self._start = time.time()
        self._last_update = 0
        self._total_width = 0

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                sys.stdout.isatty()) or
                                'ipykernel' in sys.modules or
                                'posix' in sys.modules or
                                'PYCHARM_HOSTED' in os.environ)

        self._time_after_first_step = None

    def __reset_pos(self, length):
        if self._dynamic_display:
            sys.stdout.write('\b' * length)
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')

    def update(self, current, avg_batch_metrics={}, finalize=False):  
        if current >= self.target and finalize == False:
            return

        now = time.time()
        if now - self._last_update < self.interval and not finalize:
            return
        self._last_update = now

        prev_total_width = self._total_width
        self.__reset_pos(prev_total_width)

        numdigits = int(np.log10(self.target)) + 1
        bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width - 1))
            if current < self.target:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'

        self._total_width = len(bar)
        sys.stdout.write(bar)

        if not finalize:
            time_per_unit = self._estimate_step_duration(current, now)

            eta = time_per_unit * (self.target - current)

            if eta > 3600:
                eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            info = ' - ETA: %s' % eta_format

            if avg_batch_metrics:
                info += ' - ' + ' - '.join(['{}: {:.4f}'.format(k, v) for k,v in avg_batch_metrics.items()])

            self._total_width += len(info)
            pad_count = prev_total_width - self._total_width
            if pad_count > 0:
                info += (' ' * pad_count)

            sys.stdout.write(info)
            sys.stdout.flush()

    def _estimate_step_duration(self, current, now):
        if self._time_after_first_step is not None and current > 1:
            time_per_unit = (now - self._time_after_first_step) / (current - 1)
        else:
            time_per_unit = (now - self._start) / current

        if current == 1:
            self._time_after_first_step = now
        return time_per_unit